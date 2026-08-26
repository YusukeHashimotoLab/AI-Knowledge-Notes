---
title: "第4章: ラマン分光法 (Raman Spectroscopy)"
chapter_title: "第4章: ラマン分光法 (Raman Spectroscopy)"
---

## ビデオ講義

<div class="video-container">
  <iframe
    width="560"
    height="315"
    src="https://www.youtube.com/embed/n6yPS5_-4bA"
    title="分光分析入門 第4章: ラマン分光法 (Raman Spectroscopy)"
    frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
    allowfullscreen>
  </iframe>
</div>

> このビデオは以下のテキストと同じ内容をカバーしています。お好みの学習形式をお選びください。

JP | [EN](<../../../en/MS/spectroscopy-introduction/chapter-4.html>) | Last sync: 2025-12-26

[AI寺子屋トップ](<../../index.html>)>[材料科学](<../../MS/index.html>)>[分光分析入門](<../../MS/spectroscopy-introduction/index.html>)>第4章

# 第4章: ラマン分光法 (Raman Spectroscopy)

**この章で学ぶこと:** ラマン分光法は、光と物質の非弾性散乱（ラマン散乱）を利用して分子振動情報を得る分光法です。赤外分光法と相補的な関係にあり、対称振動モードの観測に優れています。本章では、ラマン散乱の原理（ストークス散乱、アンチストークス散乱）、ラマン選択則（分極率の変化）、赤外分光法との相補性、表面増強ラマン分光法（SERS）、そして炭素材料（グラフェン、カーボンナノチューブ）への応用について学びます。Pythonを用いたスペクトル解析、D/Gバンドフィッティング、ピークデコンボリューションの実践スキルも習得します。

## 4.1 ラマン散乱の原理

### 4.1.1 光散乱の分類

物質に光を照射すると、光は様々な形で散乱されます。散乱光のエネルギー変化により、散乱は以下のように分類されます。

#### 光散乱の種類

  * **レイリー散乱（Rayleigh Scattering）:** 弾性散乱。入射光と同じエネルギー（波長）で散乱。最も強い散乱。
  * **ストークス散乱（Stokes Scattering）:** 非弾性散乱。入射光よりエネルギーが低い（波長が長い）光が散乱。分子が振動励起状態に遷移。
  * **アンチストークス散乱（Anti-Stokes Scattering）:** 非弾性散乱。入射光よりエネルギーが高い（波長が短い）光が散乱。振動励起状態から基底状態へ遷移。

**ラマンシフトの定義:**

\\[ \Delta \tilde{\nu} = \tilde{\nu}_{\text{入射}} - \tilde{\nu}_{\text{散乱}} = \frac{1}{\lambda_{\text{入射}}} - \frac{1}{\lambda_{\text{散乱}}} \\] 

ここで、\\( \tilde{\nu} \\) は波数（cm$^{-1}$）、\\( \lambda \\) は波長（cm）です。ラマンシフトは入射光の波長に依存せず、分子振動の固有エネルギーに対応します。

**波長（nm）から波数（cm$^{-1}$）への変換:**

\\[ \tilde{\nu}\,(\text{cm}^{-1}) = \frac{10^7}{\lambda\,(\text{nm})} \\] 

### 4.1.2 ラマン散乱のエネルギー図
    
    
    ```mermaid
    flowchart TD
            subgraph A[基底電子状態]
                A1[v=0 基底振動状態]
                A2[v=1 励起振動状態]
            end
    
            subgraph B[仮想状態]
                B1[仮想エネルギー準位]
            end
    
            A1 -->|入射光 hν₀| B1
            B1 -->|レイリー散乱 hν₀| A1
            B1 -->|ストークス散乱 hν₀-hνᵥ| A2
            A2 -->|入射光 hν₀| B1
            B1 -->|アンチストークス散乱 hν₀+hνᵥ| A1
    
            style A1 fill:#e3f2fd
            style A2 fill:#fff3e0
            style B1 fill:#fce4ec
        
    4.1.3 ストークス散乱とアンチストークス散乱の強度比
    アンチストークス散乱は振動励起状態の分子からのみ生じるため、その強度はストークス散乱より弱くなります。強度比はボルツマン分布に従います。
    
    ストークス/アンチストークス強度比:
            \[
            \frac{I_{\text{Anti-Stokes}}}{I_{\text{Stokes}}} = \left(\frac{\nu_0 + \nu_v}{\nu_0 - \nu_v}\right)^4 \exp\left(-\frac{h\nu_v}{k_B T}\right)
            \]
            ここで、\( \nu_0 \) は入射光の振動数、\( \nu_v \) は分子振動の振動数、\( k_B \) はボルツマン定数、\( T \) は温度です。
    室温（300 K）で振動数 1000 cm-1 の場合、アンチストークス強度はストークス強度の約 1% 程度となります。
    
    コード例1: ストークス/アンチストークス強度比の温度依存性
    import numpy as np
    import matplotlib.pyplot as plt
    
    def stokes_antistokes_ratio(wavenumber, temperature, excitation_wavelength=532):
        """
        ストークス/アンチストークス強度比を計算
    
        Parameters:
        -----------
        wavenumber : float or array
            ラマンシフト (cm^-1)
        temperature : float
            温度 (K)
        excitation_wavelength : float
            励起レーザー波長 (nm)
    
        Returns:
        --------
        ratio : float or array
            I_AntiStokes / I_Stokes
        """
        # 物理定数
        h = 6.626e-34  # J*s (プランク定数)
        c = 2.998e10   # cm/s (光速)
        k_B = 1.381e-23  # J/K (ボルツマン定数)
    
        # 入射光の波数
        nu_0 = 1e7 / excitation_wavelength  # cm^-1
    
        # 振動数項
        frequency_factor = ((nu_0 + wavenumber) / (nu_0 - wavenumber))**4
    
        # ボルツマン因子
        boltzmann_factor = np.exp(-h * c * wavenumber / (k_B * temperature))
    
        ratio = frequency_factor * boltzmann_factor
        return ratio
    
    # 温度依存性のプロット
    wavenumbers = np.array([500, 1000, 1500, 2000])  # cm^-1
    temperatures = np.linspace(100, 1000, 100)  # K
    
    plt.figure(figsize=(12, 5))
    
    # 各振動モードの強度比
    plt.subplot(1, 2, 1)
    for wn in wavenumbers:
        ratios = [stokes_antistokes_ratio(wn, T) for T in temperatures]
        plt.plot(temperatures, ratios, linewidth=2, label=f'{wn} cm$^{{-1}}$')
    
    plt.xlabel('温度 (K)', fontsize=12)
    plt.ylabel('I$_{Anti-Stokes}$ / I$_{Stokes}$', fontsize=12)
    plt.title('ストークス/アンチストークス強度比の温度依存性', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.yscale('log')
    plt.xlim(100, 1000)
    
    # 室温での振動数依存性
    plt.subplot(1, 2, 2)
    wn_range = np.linspace(100, 3000, 200)
    T_values = [200, 300, 500, 800]
    
    for T in T_values:
        ratios = stokes_antistokes_ratio(wn_range, T)
        plt.plot(wn_range, ratios, linewidth=2, label=f'{T} K')
    
    plt.xlabel('ラマンシフト (cm$^{-1}$)', fontsize=12)
    plt.ylabel('I$_{Anti-Stokes}$ / I$_{Stokes}$', fontsize=12)
    plt.title('強度比の振動数依存性', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.yscale('log')
    
    plt.tight_layout()
    plt.show()
    
    # 数値例
    print("室温(300 K)でのストークス/アンチストークス強度比:")
    for wn in [500, 1000, 1500, 2000]:
        ratio = stokes_antistokes_ratio(wn, 300)
        print(f"  {wn} cm^-1: {ratio:.4f} ({ratio*100:.2f}%)")
    
    4.2 ラマン選択則と分極率
    4.2.1 ラマン活性の条件
    ラマン散乱が観測されるためには、分子振動により分極率（polarizability）が変化する必要があります。これは赤外分光法の選択則（双極子モーメントの変化）とは異なります。
    
    ラマン選択則:
            \[
            \left(\frac{\partial \alpha}{\partial Q}\right)_{Q=0} \neq 0
            \]
            ここで、\( \alpha \) は分極率テンソル、\( Q \) は基準振動座標です。分極率が振動によって変化する場合、その振動はラマン活性です。
    誘起双極子モーメント:
            \[
            \vec{p} = \alpha \cdot \vec{E}
            \]
            分極率 \( \alpha \) は一般に3x3のテンソルであり、分子の電子雲の変形のしやすさを表します。
    
    4.2.2 ラマン活性と赤外活性の相補性
    
    相互排他則（対称中心を持つ分子）
    対称中心（反転対称性）を持つ分子では、ラマン活性と赤外活性は相互に排他的です：
    
    対称振動（偶関数）: ラマン活性、赤外不活性
    反対称振動（奇関数）: ラマン不活性、赤外活性
    
    例: CO2分子（対称中心あり）
    
    対称伸縮振動 (1388 cm-1): ラマン活性、赤外不活性
    反対称伸縮振動 (2349 cm-1): ラマン不活性、赤外活性
    変角振動 (667 cm-1): ラマン不活性、赤外活性
    
    
    
    
    
    特性
    赤外分光法 (IR)
    ラマン分光法
    
    
    
    
    観測条件
    双極子モーメントの変化
    分極率の変化
    
    
    得意な振動
    極性結合の振動
    対称振動、非極性結合
    
    
    水の影響
    強い吸収あり（測定困難）
    弱い散乱（水溶液測定可能）
    
    
    試料状態
    固体、液体、気体
    固体、液体、気体（水溶液も可）
    
    
    空間分解能
    数十マイクロメートル
    サブマイクロメートル（顕微ラマン）
    
    
    測定時間
    高速（FTIR）
    比較的長い（微弱信号）
    
    
    
    コード例2: ラマンスペクトルと赤外スペクトルの相補性シミュレーション
    import numpy as np
    import matplotlib.pyplot as plt
    
    def lorentzian_peak(x, amplitude, center, width):
        """ローレンツ関数によるピーク形状"""
        return amplitude * (width**2) / ((x - center)**2 + width**2)
    
    def simulate_ir_raman_complementarity():
        """
        CO2分子を例にIRとラマンの相補性をシミュレート
        """
        wavenumber = np.linspace(400, 2600, 2000)
    
        # CO2の振動モード
        # 対称伸縮: 1388 cm^-1 (ラマン活性、IR不活性)
        # 反対称伸縮: 2349 cm^-1 (IR活性、ラマン不活性)
        # 変角振動: 667 cm^-1 (IR活性、ラマン不活性、二重縮退)
    
        # IRスペクトル
        ir_asymmetric_stretch = lorentzian_peak(wavenumber, 0.8, 2349, 15)
        ir_bending = lorentzian_peak(wavenumber, 0.6, 667, 10)
        ir_spectrum = ir_asymmetric_stretch + ir_bending
    
        # ラマンスペクトル
        raman_symmetric_stretch = lorentzian_peak(wavenumber, 1.0, 1388, 10)
        raman_spectrum = raman_symmetric_stretch
    
        # プロット
        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
        # IRスペクトル
        axes[0].plot(wavenumber, ir_spectrum, 'b-', linewidth=2, label='IRスペクトル')
        axes[0].fill_between(wavenumber, ir_spectrum, alpha=0.3, color='blue')
        axes[0].axvline(2349, color='red', linestyle='--', alpha=0.7, label='反対称伸縮 (2349 cm$^{-1}$)')
        axes[0].axvline(667, color='green', linestyle='--', alpha=0.7, label='変角振動 (667 cm$^{-1}$)')
        axes[0].set_ylabel('吸光度', fontsize=12)
        axes[0].set_title('CO$_2$ 赤外スペクトル（IR活性モード）', fontsize=14, fontweight='bold')
        axes[0].legend(loc='upper right')
        axes[0].grid(alpha=0.3)
        axes[0].set_ylim(0, 1.2)
    
        # ラマンスペクトル
        axes[1].plot(wavenumber, raman_spectrum, 'r-', linewidth=2, label='ラマンスペクトル')
        axes[1].fill_between(wavenumber, raman_spectrum, alpha=0.3, color='red')
        axes[1].axvline(1388, color='purple', linestyle='--', alpha=0.7, label='対称伸縮 (1388 cm$^{-1}$)')
        axes[1].set_xlabel('波数 (cm$^{-1}$)', fontsize=12)
        axes[1].set_ylabel('ラマン強度', fontsize=12)
        axes[1].set_title('CO$_2$ ラマンスペクトル（ラマン活性モード）', fontsize=14, fontweight='bold')
        axes[1].legend(loc='upper right')
        axes[1].grid(alpha=0.3)
        axes[1].set_ylim(0, 1.2)
    
        plt.tight_layout()
        plt.show()
    
        # 振動モードの要約
        print("CO2分子の振動モード:")
        print("-" * 60)
        print(f"{'振動モード':<20} {'波数 (cm^-1)':<15} {'IR活性':<10} {'ラマン活性':<10}")
        print("-" * 60)
        print(f"{'対称伸縮':<20} {'1388':<15} {'不活性':<10} {'活性':<10}")
        print(f"{'反対称伸縮':<20} {'2349':<15} {'活性':<10} {'不活性':<10}")
        print(f"{'変角振動':<20} {'667':<15} {'活性':<10} {'不活性':<10}")
    
    # 実行
    simulate_ir_raman_complementarity()
    
    4.3 ラマン分光装置と測定技術
    4.3.1 ラマン分光計の構成
    
        flowchart LR
            A[励起レーザー<br/>532/633/785 nm] --> B[試料<br/>散乱光発生]
            B --> C[エッジフィルター<br/>レイリー光除去]
            C --> D[分光器<br/>波長分散]
            D --> E[CCD検出器<br/>スペクトル記録]
            E --> F[データ解析<br/>PC処理]
    
            style A fill:#e3f2fd
            style B fill:#fff3e0
            style C fill:#fce4ec
            style D fill:#e8f5e9
            style E fill:#f3e5f5
            style F fill:#ffe0b2
        
    4.3.2 励起波長の選択
    
    
    
    励起波長
    特徴
    適用例
    
    
    
    
    UV (244-325 nm)
    共鳴ラマン効果、蛍光抑制
    タンパク質、DNA
    
    
    可視 (488/514 nm)
    高効率、共鳴効果
    色素、半導体
    
    
    緑色 (532 nm)
    汎用、高効率
    炭素材料、無機材料
    
    
    赤色 (633/647 nm)
    蛍光抑制
    有機材料、高分子
    
    
    近赤外 (785/830 nm)
    蛍光抑制、生体試料
    生体組織、食品
    
    
    近赤外 (1064 nm)
    蛍光完全抑制
    蛍光物質、高分子
    
    
    
    4.4 表面増強ラマン分光法（SERS）
    4.4.1 SERSの原理
    表面増強ラマン分光法（Surface-Enhanced Raman Spectroscopy, SERS）は、金や銀などの金属ナノ構造表面で生じるラマン信号の巨大な増強効果を利用した分析手法です。増強因子は106から1014に達し、単一分子検出も可能です。
    
    SERS増強メカニズム
    
    電磁気的増強（Electromagnetic Enhancement）: 金属ナノ粒子表面のプラズモン共鳴により、局所電場が増強される。増強因子は104-108程度。
    化学的増強（Chemical Enhancement）: 分子と金属表面間の電荷移動により、分子の分極率が変化する。増強因子は102程度。
    
    総増強因子:
            \[
            \text{EF} \approx |E_{\text{local}}/E_0|^4 \times G_{\text{chemical}}
            \]
            ここで、\( E_{\text{local}} \) は局所電場、\( E_0 \) は入射電場、\( G_{\text{chemical}} \) は化学的増強因子です。
    
    4.4.2 SERS基板の種類
    
    
    
    基板タイプ
    特徴
    増強因子
    用途
    
    
    
    
    Agコロイド
    調製容易、低コスト
    106-108
    溶液中分析
    
    
    Auナノ粒子
    化学的安定、生体適合性
    105-107
    バイオセンサー
    
    
    ナノロッド配列
    高再現性、均一増強
    107-109
    定量分析
    
    
    ナノギャップ構造
    超高増強（ホットスポット）
    1010-1014
    単一分子検出
    
    
    
    コード例3: SERS増強因子のシミュレーション
    import numpy as np
    import matplotlib.pyplot as plt
    
    def sers_enhancement_factor(distance, nanoparticle_radius=50, wavelength=532):
        """
        ナノ粒子表面からの距離に対するSERS増強因子を計算
    
        Parameters:
        -----------
        distance : array
            表面からの距離 (nm)
        nanoparticle_radius : float
            ナノ粒子半径 (nm)
        wavelength : float
            励起波長 (nm)
    
        Returns:
        --------
        ef : array
            電磁気的増強因子
        """
        # 簡略化モデル: |E/E0|^4 = [(R/(R+d))^3]^4
        # R: 粒子半径, d: 表面からの距離
        R = nanoparticle_radius
    
        # 電場増強（双極子近似）
        field_enhancement = (R / (R + distance))**3
    
        # SERS増強因子（|E|^4に比例）
        ef = field_enhancement**4
    
        return ef
    
    def simulate_sers_spectrum():
        """
        通常ラマンとSERSスペクトルの比較
        """
        wavenumber = np.linspace(400, 2000, 1600)
    
        # 4-アミノチオフェノール（4-ATP）の代表的ピーク
        peak_positions = [1077, 1142, 1390, 1435, 1575]  # cm^-1
        peak_widths = [15, 12, 10, 12, 15]
        peak_intensities = [0.6, 0.4, 0.3, 0.5, 1.0]
    
        # 通常ラマンスペクトル
        normal_raman = np.zeros_like(wavenumber)
        for pos, width, intensity in zip(peak_positions, peak_widths, peak_intensities):
            normal_raman += lorentzian_peak(wavenumber, intensity, pos, width)
    
        # ノイズ追加（通常ラマンは信号が弱いためノイズが目立つ）
        normal_raman += np.random.normal(0, 0.05, len(wavenumber))
        normal_raman = np.maximum(normal_raman, 0)
    
        # SERSスペクトル（10^6倍増強、ノイズ比が大幅に改善）
        enhancement = 1e6
        sers_spectrum = normal_raman * np.sqrt(enhancement)  # 可視化のためスケール調整
        sers_spectrum += np.random.normal(0, 0.5, len(wavenumber))
    
        # プロット
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
        # 通常ラマンスペクトル
        axes[0, 0].plot(wavenumber, normal_raman, 'b-', linewidth=1.5)
        axes[0, 0].set_xlabel('ラマンシフト (cm$^{-1}$)', fontsize=12)
        axes[0, 0].set_ylabel('ラマン強度 (a.u.)', fontsize=12)
        axes[0, 0].set_title('通常ラマンスペクトル（4-ATP）', fontsize=14, fontweight='bold')
        axes[0, 0].grid(alpha=0.3)
    
        # SERSスペクトル
        axes[0, 1].plot(wavenumber, sers_spectrum, 'r-', linewidth=1.5)
        axes[0, 1].set_xlabel('ラマンシフト (cm$^{-1}$)', fontsize=12)
        axes[0, 1].set_ylabel('SERS強度 (a.u.)', fontsize=12)
        axes[0, 1].set_title('SERSスペクトル（4-ATP on Ag）', fontsize=14, fontweight='bold')
        axes[0, 1].grid(alpha=0.3)
    
        # 距離依存性
        distances = np.linspace(0.1, 30, 100)  # nm
        radii = [20, 40, 60, 80]  # nm
    
        for R in radii:
            ef = sers_enhancement_factor(distances, nanoparticle_radius=R)
            axes[1, 0].plot(distances, ef, linewidth=2, label=f'R = {R} nm')
    
        axes[1, 0].set_xlabel('表面からの距離 (nm)', fontsize=12)
        axes[1, 0].set_ylabel('増強因子 EF', fontsize=12)
        axes[1, 0].set_title('SERS増強因子の距離依存性', fontsize=14, fontweight='bold')
        axes[1, 0].legend()
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(alpha=0.3)
    
        # ホットスポットの模式図（2粒子間ギャップ）
        theta = np.linspace(0, 2*np.pi, 100)
        R = 1
        gap = 0.1
    
        # 左粒子
        x1 = R * np.cos(theta) - R - gap/2
        y1 = R * np.sin(theta)
    
        # 右粒子
        x2 = R * np.cos(theta) + R + gap/2
        y2 = R * np.sin(theta)
    
        axes[1, 1].fill(x1, y1, color='gold', alpha=0.8, label='Au nanoparticle')
        axes[1, 1].fill(x2, y2, color='gold', alpha=0.8)
        axes[1, 1].scatter([0], [0], s=200, c='red', marker='*', zorder=5, label='Hot spot')
        axes[1, 1].set_xlim(-3, 3)
        axes[1, 1].set_ylim(-2, 2)
        axes[1, 1].set_aspect('equal')
        axes[1, 1].set_xlabel('x (a.u.)', fontsize=12)
        axes[1, 1].set_ylabel('y (a.u.)', fontsize=12)
        axes[1, 1].set_title('ナノギャップ ホットスポット', fontsize=14, fontweight='bold')
        axes[1, 1].legend(loc='upper right')
        axes[1, 1].grid(alpha=0.3)
    
        plt.tight_layout()
        plt.show()
    
    # 実行
    simulate_sers_spectrum()
    
    4.5 炭素材料のラマン分光分析
    4.5.1 炭素材料の特徴的ラマンバンド
    炭素材料（グラファイト、グラフェン、カーボンナノチューブ、ダイヤモンドなど）はラマン分光法で特徴的なスペクトルを示し、構造評価に広く利用されています。
    
    炭素材料の主要ラマンバンド
    
    Gバンド（G band）: 約1580 cm-1。sp2炭素のE2g面内伸縮振動。グラフェン、グラファイト、CNTで観測。
    Dバンド（D band）: 約1350 cm-1。欠陥活性化モード（disorder-induced）。結晶欠陥、エッジ、sp3炭素で活性化。
    2Dバンド（2D/G'band）: 約2700 cm-1。Dバンドの倍音。グラフェン層数の評価に重要。
    D'バンド（D' band）: 約1620 cm-1。欠陥活性化モード。Gバンドの肩に現れる。
    RBM（Radial Breathing Mode）: 100-300 cm-1。単層カーボンナノチューブ特有の径方向振動。
    
    
    
    
    
    炭素材料
    Gバンド位置 (cm-1)
    D/G比
    2D/G比
    特徴
    
    
    
    
    単層グラフェン
    1580-1590
    0（欠陥なし）
    2-4
    2Dがシャープな単一ピーク
    
    
    二層グラフェン
    1580-1590
    0（欠陥なし）
    1-2
    2Dが4成分に分裂
    
    
    HOPG（グラファイト）
    1580
    0（欠陥なし）
    0.5-1
    2Dが2成分
    
    
    単層CNT
    1590-1600
    0.1-0.3
    0.5-1
    RBMあり、Gバンド分裂
    
    
    多層CNT
    1580-1590
    0.2-1.0
    0.3-0.8
    RBM弱い
    
    
    アモルファスカーボン
    1500-1600
    1.0-2.0
    なし/微弱
    ブロードなピーク
    
    
    ダイヤモンド
    1332
    -
    -
    単一シャープピーク
    
    
    
    4.5.2 D/G比による欠陥評価
    
    D/G強度比と欠陥密度の関係（Tuinstra-Koenig式）:
            \[
            \frac{I_D}{I_G} = \frac{C(\lambda_L)}{L_a}
            \]
            ここで、\( L_a \) は結晶子サイズ（面内相関長）、\( C(\lambda_L) \) は励起波長依存の定数です。
    Canado式（ナノ結晶グラファイト）:
            \[
            L_a\,(\text{nm}) = (2.4 \times 10^{-10}) \lambda_L^4 \left(\frac{I_D}{I_G}\right)^{-1}
            \]
            ここで、\( \lambda_L \) は励起レーザー波長（nm）です。
    
    コード例4: グラフェンのラマンスペクトル解析とD/G比計算
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.optimize import curve_fit
    from scipy.signal import find_peaks
    
    def gaussian_peak(x, amplitude, center, width):
        """ガウス関数によるピーク形状"""
        return amplitude * np.exp(-((x - center)**2) / (2 * width**2))
    
    def multi_peak_model(x, *params):
        """
        複数ピークモデル（Dバンド、Gバンド、D'バンド、2Dバンド）
        params: [A_D, c_D, w_D, A_G, c_G, w_G, A_D', c_D', w_D', A_2D, c_2D, w_2D]
        """
        n_peaks = len(params) // 3
        result = np.zeros_like(x)
        for i in range(n_peaks):
            A, c, w = params[3*i:3*i+3]
            result += gaussian_peak(x, A, c, w)
        return result
    
    def analyze_graphene_raman(wavenumber, intensity, peak_names=['D', 'G', "D'", '2D']):
        """
        グラフェンラマンスペクトルの解析
    
        Parameters:
        -----------
        wavenumber : array
            ラマンシフト (cm^-1)
        intensity : array
            ラマン強度
        peak_names : list
            フィットするピーク名
    
        Returns:
        --------
        results : dict
            各ピークのパラメータと評価指標
        """
        # 初期推定値（グラフェンの典型値）
        initial_params = {
            'D': [100, 1350, 30],      # [振幅, 中心, 幅]
            'G': [500, 1580, 15],
            "D'": [50, 1620, 10],
            '2D': [400, 2700, 25]
        }
    
        # フィッティング用パラメータ
        p0 = []
        for name in peak_names:
            p0.extend(initial_params[name])
    
        # バウンド設定
        lower_bounds = []
        upper_bounds = []
        for name in peak_names:
            center = initial_params[name][1]
            lower_bounds.extend([0, center - 50, 5])
            upper_bounds.extend([np.max(intensity)*2, center + 50, 100])
    
        try:
            popt, pcov = curve_fit(multi_peak_model, wavenumber, intensity,
                                   p0=p0, bounds=(lower_bounds, upper_bounds), maxfev=10000)
        except RuntimeError:
            print("フィッティングが収束しませんでした。")
            return None
    
        # 結果の整理
        results = {'peaks': {}}
        for i, name in enumerate(peak_names):
            A, c, w = popt[3*i:3*i+3]
            # ピーク面積（ガウスの場合: A * sqrt(2*pi) * w）
            area = A * np.sqrt(2 * np.pi) * w
            results['peaks'][name] = {
                'amplitude': A,
                'center': c,
                'width_fwhm': 2.355 * w,  # FWHM = 2.355 * sigma
                'area': area
            }
    
        # D/G比の計算
        if 'D' in results['peaks'] and 'G' in results['peaks']:
            results['D_G_ratio'] = results['peaks']['D']['area'] / results['peaks']['G']['area']
            results['D_G_intensity_ratio'] = results['peaks']['D']['amplitude'] / results['peaks']['G']['amplitude']
    
        # 2D/G比の計算
        if '2D' in results['peaks'] and 'G' in results['peaks']:
            results['2D_G_ratio'] = results['peaks']['2D']['area'] / results['peaks']['G']['area']
            results['2D_G_intensity_ratio'] = results['peaks']['2D']['amplitude'] / results['peaks']['G']['amplitude']
    
        # フィッティング曲線
        results['fitted_curve'] = multi_peak_model(wavenumber, *popt)
        results['popt'] = popt
    
        return results
    
    def simulate_graphene_spectrum(defect_level='low'):
        """
        グラフェンラマンスペクトルのシミュレーション
    
        Parameters:
        -----------
        defect_level : str
            'pristine', 'low', 'medium', 'high'
        """
        wavenumber = np.linspace(1100, 3100, 2000)
    
        # 欠陥レベルに応じたD/G比
        defect_params = {
            'pristine': {'D': 0, 'G': 500, 'D_prime': 0, '2D': 1200},
            'low': {'D': 50, 'G': 500, 'D_prime': 20, '2D': 1000},
            'medium': {'D': 200, 'G': 500, 'D_prime': 80, '2D': 600},
            'high': {'D': 400, 'G': 500, 'D_prime': 150, '2D': 300}
        }
    
        params = defect_params[defect_level]
    
        # スペクトル生成
        D_band = gaussian_peak(wavenumber, params['D'], 1350, 25)
        G_band = gaussian_peak(wavenumber, params['G'], 1580, 12)
        D_prime = gaussian_peak(wavenumber, params['D_prime'], 1620, 10)
        two_D = gaussian_peak(wavenumber, params['2D'], 2700, 25)
    
        spectrum = D_band + G_band + D_prime + two_D
        noise = np.random.normal(0, 10, len(wavenumber))
        spectrum_noisy = spectrum + noise
    
        return wavenumber, spectrum_noisy
    
    # 異なる欠陥レベルのグラフェンスペクトルを比較
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    defect_levels = ['pristine', 'low', 'medium', 'high']
    titles = ['高品質グラフェン', '低欠陥グラフェン', '中欠陥グラフェン', '高欠陥グラフェン']
    
    for ax, level, title in zip(axes.flatten(), defect_levels, titles):
        wavenumber, spectrum = simulate_graphene_spectrum(level)
    
        ax.plot(wavenumber, spectrum, 'b-', linewidth=1.5)
        ax.axvline(1350, color='red', linestyle='--', alpha=0.5, label='D band')
        ax.axvline(1580, color='green', linestyle='--', alpha=0.5, label='G band')
        ax.axvline(2700, color='purple', linestyle='--', alpha=0.5, label='2D band')
        ax.set_xlabel('ラマンシフト (cm$^{-1}$)', fontsize=11)
        ax.set_ylabel('ラマン強度 (a.u.)', fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # 定量解析の例
    print("グラフェンラマンスペクトル解析結果:")
    print("-" * 60)
    for level, title in zip(defect_levels, titles):
        wavenumber, spectrum = simulate_graphene_spectrum(level)
        results = analyze_graphene_raman(wavenumber, spectrum)
        if results:
            print(f"\n{title}:")
            print(f"  D/G強度比: {results.get('D_G_intensity_ratio', 0):.3f}")
            print(f"  2D/G強度比: {results.get('2D_G_intensity_ratio', 0):.3f}")
    
    4.5.3 カーボンナノチューブのRBMモード
    単層カーボンナノチューブ（SWCNT）は、径方向呼吸振動（Radial Breathing Mode, RBM）と呼ばれる特徴的な低波数モードを示します。RBM振動数からナノチューブの直径を決定できます。
    
    RBM振動数と直径の関係:
            \[
            \omega_{\text{RBM}} = \frac{A}{d_t} + B
            \]
            ここで、\( \omega_{\text{RBM}} \) はRBM振動数（cm-1）、\( d_t \) はナノチューブ直径（nm）です。
    典型的なパラメータ: \( A = 234 \) cm-1nm, \( B = 10 \) cm-1（基板効果による補正）
    直径の計算式:
            \[
            d_t = \frac{A}{\omega_{\text{RBM}} - B}
            \]
        
    コード例5: CNTのRBMモード解析と直径決定
    import numpy as np
    import matplotlib.pyplot as plt
    
    def rbm_to_diameter(omega_rbm, A=234, B=10):
        """
        RBM振動数からCNT直径を計算
    
        Parameters:
        -----------
        omega_rbm : float or array
            RBM振動数 (cm^-1)
        A : float
            定数 (cm^-1 nm)、デフォルト 234
        B : float
            定数 (cm^-1)、デフォルト 10
    
        Returns:
        --------
        diameter : float or array
            CNT直径 (nm)
        """
        return A / (omega_rbm - B)
    
    def diameter_to_rbm(diameter, A=234, B=10):
        """
        CNT直径からRBM振動数を計算
        """
        return A / diameter + B
    
    def chirality_to_diameter(n, m, a_cc=0.142):
        """
        カイラリティ(n, m)からCNT直径を計算
    
        Parameters:
        -----------
        n, m : int
            カイラル指数
        a_cc : float
            C-C結合長 (nm)、デフォルト 0.142 nm
    
        Returns:
        --------
        diameter : float
            CNT直径 (nm)
        """
        a = np.sqrt(3) * a_cc  # 格子定数
        diameter = a * np.sqrt(n**2 + n*m + m**2) / np.pi
        return diameter
    
    def simulate_swcnt_raman():
        """
        SWCNTのRBMスペクトルをシミュレート
        """
        wavenumber_rbm = np.linspace(100, 350, 500)
        wavenumber_high = np.linspace(1200, 1700, 500)
    
        # 混合サンプル中の異なる直径のSWCNT
        # (n, m)カイラリティの例
        chiralities = [(10, 10), (8, 7), (6, 5), (11, 0)]
    
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
        # RBMスペクトル
        rbm_spectrum = np.zeros_like(wavenumber_rbm)
        cnt_info = []
    
        for n, m in chiralities:
            d = chirality_to_diameter(n, m)
            omega = diameter_to_rbm(d)
            # 半導体か金属かを判定
            mod_value = (n - m) % 3
            if mod_value == 0:
                cnt_type = '金属'
                color = 'red'
            else:
                cnt_type = '半導体'
                color = 'blue'
    
            amplitude = np.random.uniform(0.5, 1.0)
            rbm_spectrum += gaussian_peak(wavenumber_rbm, amplitude, omega, 3)
            cnt_info.append({
                'chirality': (n, m),
                'diameter': d,
                'rbm': omega,
                'type': cnt_type
            })
    
        noise = np.random.normal(0, 0.02, len(wavenumber_rbm))
        rbm_spectrum += noise
    
        axes[0].plot(wavenumber_rbm, rbm_spectrum, 'b-', linewidth=1.5)
        for info in cnt_info:
            axes[0].axvline(info['rbm'], color='red' if info['type']=='金属' else 'blue',
                           linestyle='--', alpha=0.7)
            axes[0].text(info['rbm'], 1.1, f"({info['chirality'][0]},{info['chirality'][1]})",
                        rotation=90, fontsize=9, ha='center')
    
        axes[0].set_xlabel('ラマンシフト (cm$^{-1}$)', fontsize=12)
        axes[0].set_ylabel('ラマン強度 (a.u.)', fontsize=12)
        axes[0].set_title('SWCNT RBMスペクトル', fontsize=14, fontweight='bold')
        axes[0].grid(alpha=0.3)
        axes[0].set_ylim(0, 1.5)
    
        # 高波数領域（G+, G-）
        G_plus = gaussian_peak(wavenumber_high, 1.0, 1590, 10)
        G_minus = gaussian_peak(wavenumber_high, 0.4, 1570, 15)  # 金属CNTでブロード
        D_band = gaussian_peak(wavenumber_high, 0.15, 1350, 25)
        high_spectrum = G_plus + G_minus + D_band + np.random.normal(0, 0.02, len(wavenumber_high))
    
        axes[1].plot(wavenumber_high, high_spectrum, 'g-', linewidth=1.5)
        axes[1].axvline(1590, color='red', linestyle='--', alpha=0.7, label='G$^+$')
        axes[1].axvline(1570, color='blue', linestyle='--', alpha=0.7, label='G$^-$')
        axes[1].axvline(1350, color='purple', linestyle='--', alpha=0.7, label='D band')
        axes[1].set_xlabel('ラマンシフト (cm$^{-1}$)', fontsize=12)
        axes[1].set_ylabel('ラマン強度 (a.u.)', fontsize=12)
        axes[1].set_title('SWCNT D/Gバンド', fontsize=14, fontweight='bold')
        axes[1].legend()
        axes[1].grid(alpha=0.3)
    
        # RBM-直径関係
        diameters = np.linspace(0.5, 2.5, 100)
        rbm_frequencies = diameter_to_rbm(diameters)
    
        axes[2].plot(diameters, rbm_frequencies, 'b-', linewidth=2)
        for info in cnt_info:
            color = 'red' if info['type'] == '金属' else 'blue'
            axes[2].scatter(info['diameter'], info['rbm'], s=100, c=color,
                           edgecolor='black', linewidth=1.5, zorder=5)
            axes[2].annotate(f"({info['chirality'][0]},{info['chirality'][1]})",
                            (info['diameter'], info['rbm']),
                            textcoords="offset points", xytext=(10, 5), fontsize=9)
    
        axes[2].set_xlabel('CNT直径 (nm)', fontsize=12)
        axes[2].set_ylabel('RBM振動数 (cm$^{-1}$)', fontsize=12)
        axes[2].set_title('RBM振動数と直径の関係', fontsize=14, fontweight='bold')
        axes[2].grid(alpha=0.3)
    
        plt.tight_layout()
        plt.show()
    
        # 結果表示
        print("検出されたSWCNT:")
        print("-" * 60)
        print(f"{'カイラリティ':<15} {'直径 (nm)':<12} {'RBM (cm^-1)':<12} {'タイプ':<10}")
        print("-" * 60)
        for info in cnt_info:
            print(f"({info['chirality'][0]:2d},{info['chirality'][1]:2d}){'':<10} "
                  f"{info['diameter']:<12.3f} {info['rbm']:<12.1f} {info['type']:<10}")
    
    # 実行
    simulate_swcnt_raman()
    
    4.5.4 シリコンの結晶性評価
    同じピーク形状の考え方は炭素材料以外にも適用できます。結晶シリコンは 520 cm^-1 に半値幅
    わずか 3-4 cm^-1 の鋭い一次光学フォノンピークを示しますが、アモルファスシリコンでは長距離
    秩序が失われて q ~ 0 の選択則が緩和されるため、480 cm^-1 付近を中心とする数十 cm^-1 幅の
    ブロードなバンドに変化します。混相膜では両者が重畳するため、400-560 cm^-1 の包絡線を
    結晶成分と非晶質成分に分離することで結晶化度を見積もれます:
            \[
            X_c \approx \frac{I_c}{I_c + I_a}
            \]
            ここで \( I_c \)、\( I_a \) はそれぞれ 520 cm^-1 成分と 480 cm^-1 成分の積分強度です。
    ただしこの式は簡略化されたものです。定量評価では粒界に対応する 500-510 cm^-1 付近の
    中間成分を加えた3成分フィッティングを行い、さらにラマン散乱断面積比
    \( y = \sigma_a/\sigma_c \)（およそ 0.7-0.9、粒径と励起波長に依存）で補正します。
    したがって上式の X_c は絶対的な結晶体積分率ではなく、校正を前提とした相対指標として
    扱うべきです。またレーザー加熱は 520 cm^-1 ピークを低波数側にシフトさせ広幅化するため、
    ピーク位置が安定する程度に励起パワーを抑える必要があります。
    
    4.5.5 生体・医療応用
    水のラマン散乱が弱いことから、ラマン分光は生体組織や生細胞を無染色・無標識のまま直接
    測定できます。タンパク質や脂質は特徴的なバンドを示し、フェニルアラニンの環呼吸振動
    1003 cm^-1、アミドIII 1250 cm^-1 付近、CH2 変角 1445 cm^-1 付近、アミドI 1660 cm^-1
    付近の相対強度の変化から病変組織と正常組織を識別できます。これがラマン分光による
    がん組織診断の原理です。共焦点ラマンイメージングはこれらのバンドをサブミクロン分解能で
    マッピングする無標識イメージングを可能にし、時間分解測定では細胞内の薬剤自身の
    ラマンシグネチャの変化から薬剤と細胞の相互作用を追跡できます。これらの応用でも、
    非破壊・最小限の前処理・ガラスや水環境との適合性というラマン分光の実用上の利点が
    そのまま生きています。
    
    4.6 ラマンスペクトルの前処理とピークフィッティング
    4.6.1 ベースライン補正
    ラマンスペクトルには、蛍光バックグラウンドや装置由来のベースラインが含まれることがあります。定量解析の前にベースライン補正が必要です。
    コード例6: ベースライン補正とスペクトル前処理
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.signal import savgol_filter
    from scipy import sparse
    from scipy.sparse.linalg import spsolve
    
    def baseline_als(y, lam=1e5, p=0.01, niter=10):
        """
        非対称最小二乗法によるベースライン推定（ALS法）
    
        Parameters:
        -----------
        y : array
            入力スペクトル
        lam : float
            平滑化パラメータ（大きいほど滑らか）
        p : float
            非対称パラメータ（0 < p < 1）
        niter : int
            反復回数
    
        Returns:
        --------
        baseline : array
            推定されたベースライン
        """
        L = len(y)
        D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L, L-2))
        D = lam * D.dot(D.T)
        w = np.ones(L)
        W = sparse.spdiags(w, 0, L, L)
    
        for i in range(niter):
            W.setdiag(w)
            Z = W + D
            baseline = spsolve(Z, w * y)
            w = p * (y > baseline) + (1 - p) * (y < baseline)
    
        return baseline
    
    def preprocess_raman_spectrum(wavenumber, raw_spectrum, smooth_window=11, smooth_order=3):
        """
        ラマンスペクトルの前処理パイプライン
    
        Parameters:
        -----------
        wavenumber : array
            ラマンシフト
        raw_spectrum : array
            生スペクトル
        smooth_window : int
            Savitzky-Golayフィルタの窓幅
        smooth_order : int
            多項式次数
    
        Returns:
        --------
        processed : dict
            前処理後のスペクトルと中間結果
        """
        # 1. スムージング
        smoothed = savgol_filter(raw_spectrum, smooth_window, smooth_order)
    
        # 2. ベースライン推定（ALS法）
        baseline = baseline_als(smoothed, lam=1e6, p=0.001)
    
        # 3. ベースライン補正
        corrected = smoothed - baseline
    
        # 4. 正規化（最大値で規格化）
        normalized = corrected / np.max(corrected)
    
        return {
            'wavenumber': wavenumber,
            'raw': raw_spectrum,
            'smoothed': smoothed,
            'baseline': baseline,
            'corrected': corrected,
            'normalized': normalized
        }
    
    def demonstrate_preprocessing():
        """
        前処理の効果をデモンストレーション
        """
        # 蛍光バックグラウンドを含むシミュレーションデータ
        wavenumber = np.linspace(800, 2000, 1200)
    
        # 真のラマンピーク（グラファイト様）
        D_band = gaussian_peak(wavenumber, 100, 1350, 30)
        G_band = gaussian_peak(wavenumber, 300, 1580, 15)
    
        # 蛍光バックグラウンド（広い帯域の曲線）
        fluorescence = 50 * np.exp(-((wavenumber - 1400)**2) / (2 * 300**2))
    
        # 傾斜ベースライン
        slope_baseline = 0.05 * (wavenumber - 800)
    
        # 合成スペクトル
        raw = D_band + G_band + fluorescence + slope_baseline
    
        # ノイズ追加
        noise = np.random.normal(0, 5, len(wavenumber))
        raw_noisy = raw + noise
    
        # 前処理実行
        result = preprocess_raman_spectrum(wavenumber, raw_noisy)
    
        # プロット
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
        # 生データとスムージング
        axes[0, 0].plot(wavenumber, raw_noisy, 'gray', alpha=0.5, linewidth=0.5, label='生データ')
        axes[0, 0].plot(wavenumber, result['smoothed'], 'b-', linewidth=1.5, label='スムージング後')
        axes[0, 0].set_xlabel('ラマンシフト (cm$^{-1}$)', fontsize=12)
        axes[0, 0].set_ylabel('強度 (a.u.)', fontsize=12)
        axes[0, 0].set_title('Step 1: ノイズ除去（Savitzky-Golay）', fontsize=14, fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.3)
    
        # ベースライン推定
        axes[0, 1].plot(wavenumber, result['smoothed'], 'b-', linewidth=1.5, label='スムージング後')
        axes[0, 1].plot(wavenumber, result['baseline'], 'r-', linewidth=2, label='推定ベースライン')
        axes[0, 1].set_xlabel('ラマンシフト (cm$^{-1}$)', fontsize=12)
        axes[0, 1].set_ylabel('強度 (a.u.)', fontsize=12)
        axes[0, 1].set_title('Step 2: ベースライン推定（ALS法）', fontsize=14, fontweight='bold')
        axes[0, 1].legend()
        axes[0, 1].grid(alpha=0.3)
    
        # ベースライン補正後
        axes[1, 0].plot(wavenumber, result['corrected'], 'g-', linewidth=1.5)
        axes[1, 0].axhline(0, color='black', linestyle='--', alpha=0.5)
        axes[1, 0].set_xlabel('ラマンシフト (cm$^{-1}$)', fontsize=12)
        axes[1, 0].set_ylabel('強度 (a.u.)', fontsize=12)
        axes[1, 0].set_title('Step 3: ベースライン補正後', fontsize=14, fontweight='bold')
        axes[1, 0].grid(alpha=0.3)
    
        # 正規化後
        axes[1, 1].plot(wavenumber, result['normalized'], 'purple', linewidth=1.5)
        axes[1, 1].axvline(1350, color='red', linestyle='--', alpha=0.7, label='D band')
        axes[1, 1].axvline(1580, color='green', linestyle='--', alpha=0.7, label='G band')
        axes[1, 1].set_xlabel('ラマンシフト (cm$^{-1}$)', fontsize=12)
        axes[1, 1].set_ylabel('正規化強度', fontsize=12)
        axes[1, 1].set_title('Step 4: 正規化（最大値 = 1）', fontsize=14, fontweight='bold')
        axes[1, 1].legend()
        axes[1, 1].grid(alpha=0.3)
    
        plt.tight_layout()
        plt.show()
    
    # 実行
    demonstrate_preprocessing()
    
    4.6.2 ピークデコンボリューション
    重なり合ったピークを分離するため、複数のピーク関数（ガウス、ローレンツ、Voigt）の和でスペクトルをフィッティングします。
    コード例7: D/Gバンドの詳細フィッティング
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.optimize import curve_fit, minimize
    from scipy.special import wofz
    
    def voigt_peak(x, amplitude, center, sigma, gamma):
        """
        Voigt関数（ガウスとローレンツの畳み込み）
    
        Parameters:
        -----------
        x : array
            波数
        amplitude : float
            ピーク振幅
        center : float
            ピーク中心
        sigma : float
            ガウス幅
        gamma : float
            ローレンツ幅
    
        Returns:
        --------
        y : array
            Voigt関数の値
        """
        z = ((x - center) + 1j*gamma) / (sigma * np.sqrt(2))
        return amplitude * np.real(wofz(z)) / (sigma * np.sqrt(2*np.pi))
    
    def five_peak_model(x, *params):
        """
        5ピークモデル（D1, D2, D3, D4, G）
        炭素材料の詳細解析用
    
        params: [A1, c1, w1, A2, c2, w2, ...]
        """
        n_peaks = len(params) // 3
        result = np.zeros_like(x)
        for i in range(n_peaks):
            A, c, w = params[3*i:3*i+3]
            result += gaussian_peak(x, A, c, w)
        return result
    
    def fit_carbon_dg_bands(wavenumber, intensity, model='five_peak'):
        """
        炭素材料のD/Gバンド領域のフィッティング
    
        Parameters:
        -----------
        wavenumber : array
            ラマンシフト (1000-1800 cm^-1 の範囲が望ましい)
        intensity : array
            ラマン強度
        model : str
            'two_peak' (D, G only) または 'five_peak' (D1, D2, D3, D4, G)
    
        Returns:
        --------
        result : dict
            フィッティング結果
        """
        if model == 'two_peak':
            # 2ピークモデル（D, G）
            peak_names = ['D', 'G']
            p0 = [np.max(intensity)*0.3, 1350, 50, np.max(intensity), 1580, 20]
            bounds = ([0, 1300, 10, 0, 1550, 5],
                      [np.max(intensity)*2, 1400, 100, np.max(intensity)*2, 1620, 50])
        else:
            # 5ピークモデル（D1, D2, D3, D4, G）
            # D1: ~1350 (主欠陥ピーク)
            # D2/D': ~1620 (欠陥ピーク)
            # D3: ~1500 (アモルファス炭素)
            # D4: ~1200 (sp3炭素/多重フォノン)
            # G: ~1580 (sp2 in-plane)
            peak_names = ['D4', 'D1', 'D3', 'G', 'D2']
            I_max = np.max(intensity)
            p0 = [I_max*0.1, 1200, 50,   # D4
                  I_max*0.4, 1350, 40,   # D1
                  I_max*0.1, 1500, 80,   # D3
                  I_max*0.8, 1580, 20,   # G
                  I_max*0.2, 1620, 15]   # D2
    
            bounds = ([0, 1150, 20, 0, 1300, 20, 0, 1450, 30, 0, 1560, 10, 0, 1600, 5],
                      [I_max*2]*15)
            bounds[1][2] = 1250   # D4 center max
            bounds[1][5] = 1400   # D1 center max
            bounds[1][8] = 1560   # D3 center max
            bounds[1][11] = 1600  # G center max
            bounds[1][14] = 1650  # D2 center max
    
        n_peaks = len(peak_names)
    
        try:
            popt, pcov = curve_fit(
                lambda x, *p: five_peak_model(x, *p) if len(p) == 15
                              else multi_peak_model(x, *p),
                wavenumber, intensity, p0=p0, bounds=bounds, maxfev=20000
            )
            perr = np.sqrt(np.diag(pcov))
        except RuntimeError:
            print("Warning: フィッティングが収束しませんでした。")
            return None
    
        # 結果の整理
        result = {'peaks': {}, 'total_fit': None}
        total_area = 0
    
        for i, name in enumerate(peak_names):
            A, c, w = popt[3*i:3*i+3]
            area = A * np.sqrt(2*np.pi) * w
            total_area += area
            result['peaks'][name] = {
                'amplitude': A,
                'center': c,
                'width': w,
                'fwhm': 2.355 * w,
                'area': area
            }
    
        # 面積比の計算
        D_total = sum([result['peaks'][k]['area'] for k in result['peaks'] if k.startswith('D')])
        G_area = result['peaks']['G']['area']
        result['D_G_area_ratio'] = D_total / G_area
        result['D1_G_ratio'] = result['peaks'].get('D1', result['peaks'].get('D', {})).get('area', 0) / G_area
    
        # フィット曲線
        result['total_fit'] = five_peak_model(wavenumber, *popt) if len(popt) == 15 else multi_peak_model(wavenumber, *popt)
        result['popt'] = popt
        result['peak_names'] = peak_names
    
        return result
    
    def demonstrate_dg_fitting():
        """
        D/Gバンドフィッティングのデモンストレーション
        """
        # シミュレーションデータ（炭素材料）
        wavenumber = np.linspace(1000, 1800, 800)
    
        # 5成分スペクトル
        D4 = gaussian_peak(wavenumber, 30, 1210, 50)
        D1 = gaussian_peak(wavenumber, 150, 1350, 45)
        D3 = gaussian_peak(wavenumber, 40, 1510, 70)
        G = gaussian_peak(wavenumber, 300, 1585, 18)
        D2 = gaussian_peak(wavenumber, 80, 1620, 15)
    
        spectrum = D4 + D1 + D3 + G + D2
        noise = np.random.normal(0, 8, len(wavenumber))
        spectrum_noisy = spectrum + noise
        spectrum_noisy = np.maximum(spectrum_noisy, 0)
    
        # フィッティング
        result = fit_carbon_dg_bands(wavenumber, spectrum_noisy, model='five_peak')
    
        if result is None:
            print("フィッティング失敗")
            return
    
        # プロット
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
        # フィッティング結果
        axes[0].plot(wavenumber, spectrum_noisy, 'ko', markersize=2, alpha=0.3, label='実測データ')
        axes[0].plot(wavenumber, result['total_fit'], 'r-', linewidth=2, label='フィット（合計）')
    
        # 各成分を表示
        colors = ['blue', 'green', 'cyan', 'magenta', 'orange']
        for i, name in enumerate(result['peak_names']):
            peak = result['peaks'][name]
            component = gaussian_peak(wavenumber, peak['amplitude'], peak['center'], peak['width'])
            axes[0].fill_between(wavenumber, component, alpha=0.3, color=colors[i], label=f'{name} ({peak["center"]:.0f} cm$^{{-1}}$)')
    
        axes[0].set_xlabel('ラマンシフト (cm$^{-1}$)', fontsize=12)
        axes[0].set_ylabel('ラマン強度 (a.u.)', fontsize=12)
        axes[0].set_title('D/Gバンド 5ピークフィッティング', fontsize=14, fontweight='bold')
        axes[0].legend(loc='upper left', fontsize=9)
        axes[0].grid(alpha=0.3)
    
        # 残差
        residual = spectrum_noisy - result['total_fit']
        axes[1].plot(wavenumber, residual, 'b-', linewidth=1)
        axes[1].axhline(0, color='red', linestyle='--')
        axes[1].set_xlabel('ラマンシフト (cm$^{-1}$)', fontsize=12)
        axes[1].set_ylabel('残差', fontsize=12)
        axes[1].set_title('フィッティング残差', fontsize=14, fontweight='bold')
        axes[1].grid(alpha=0.3)
    
        plt.tight_layout()
        plt.show()
    
        # 結果表示
        print("\nフィッティング結果:")
        print("-" * 70)
        print(f"{'ピーク':<8} {'中心 (cm^-1)':<15} {'FWHM (cm^-1)':<15} {'面積 (%)':<15}")
        print("-" * 70)
    
        total_area = sum([p['area'] for p in result['peaks'].values()])
        for name, peak in result['peaks'].items():
            area_pct = peak['area'] / total_area * 100
            print(f"{name:<8} {peak['center']:<15.1f} {peak['fwhm']:<15.1f} {area_pct:<15.1f}")
    
        print("-" * 70)
        print(f"D/G面積比 (D1/G): {result['D1_G_ratio']:.3f}")
        print(f"D/G面積比 (全D/G): {result['D_G_area_ratio']:.3f}")
    
    # 実行
    demonstrate_dg_fitting()
    
    4.7 演習問題
    
    基礎問題（Easy）
    問題1: ラマンシフトの計算
    532 nmの励起レーザーを使用したとき、ストークス散乱光が 545 nm で観測された。ラマンシフト（cm-1）を計算せよ。
    
    解答を見る
    
    解答:
    波数への変換:
                    \[
                    \tilde{\nu}_{\text{入射}} = \frac{10^7}{532} = 18797\,\text{cm}^{-1}
                    \]
                    \[
                    \tilde{\nu}_{\text{散乱}} = \frac{10^7}{545} = 18349\,\text{cm}^{-1}
                    \]
                    ラマンシフト:
                    \[
                    \Delta\tilde{\nu} = 18797 - 18349 = 448\,\text{cm}^{-1}
                    \]
                    答え: 448 cm-1
    Pythonコード:
    lambda_excitation = 532  # nm
    lambda_scattered = 545   # nm
    
    nu_excitation = 1e7 / lambda_excitation  # cm^-1
    nu_scattered = 1e7 / lambda_scattered    # cm^-1
    
    raman_shift = nu_excitation - nu_scattered
    print(f"ラマンシフト: {raman_shift:.0f} cm^-1")
    
    
    
    問題2: ストークス/アンチストークス強度比
    室温（300 K）で 1000 cm-1 のラマンバンドにおけるストークス散乱とアンチストークス散乱の強度比を計算せよ。振動数項は無視してよい。
    
    解答を見る
    
    解答:
    ボルツマン因子のみを考慮:
                    \[
                    \frac{I_{\text{AS}}}{I_{\text{S}}} = \exp\left(-\frac{hc\tilde{\nu}}{k_B T}\right)
                    \]
                    計算:
                    \[
                    \frac{hc\tilde{\nu}}{k_B T} = \frac{6.626 \times 10^{-34} \times 3 \times 10^{10} \times 1000}{1.381 \times 10^{-23} \times 300} = 4.80
                    \]
                    \[
                    \frac{I_{\text{AS}}}{I_{\text{S}}} = e^{-4.80} = 0.0082 \approx 0.8\%
                    \]
                    答え: 約 0.8%（アンチストークスはストークスの約100分の1）
    Pythonコード:
    import numpy as np
    
    h = 6.626e-34  # J*s
    c = 3e10       # cm/s
    k_B = 1.381e-23  # J/K
    T = 300        # K
    nu = 1000      # cm^-1
    
    ratio = np.exp(-h * c * nu / (k_B * T))
    print(f"I_AS / I_S = {ratio:.4f} ({ratio*100:.2f}%)")
    
    
    
    問題3: CNT直径の決定
    単層カーボンナノチューブのRBMモードが 180 cm-1 に観測された。直径を計算せよ（A = 234 cm-1nm, B = 10 cm-1）。
    
    解答を見る
    
    解答:
                    \[
                    d_t = \frac{A}{\omega_{\text{RBM}} - B} = \frac{234}{180 - 10} = \frac{234}{170} = 1.38\,\text{nm}
                    \]
                    答え: 約 1.38 nm
    Pythonコード:
    A = 234  # cm^-1 nm
    B = 10   # cm^-1
    omega_rbm = 180  # cm^-1
    
    diameter = A / (omega_rbm - B)
    print(f"CNT直径: {diameter:.2f} nm")
    
    
    
    
    
    中級問題（Medium）
    問題4: D/G比による結晶子サイズ評価
    グラファイト試料のラマンスペクトルで、D/G強度比が 0.5 であった。Canado式を用いて結晶子サイズ \( L_a \) を推定せよ（励起波長 514 nm）。
    
    解答を見る
    
    解答:
    Canado式:
                    \[
                    L_a = (2.4 \times 10^{-10}) \lambda_L^4 \left(\frac{I_D}{I_G}\right)^{-1}
                    \]
                    計算:
                    \[
                    L_a = (2.4 \times 10^{-10}) \times 514^4 \times \frac{1}{0.5}
                    \]
                    \[
                    L_a = (2.4 \times 10^{-10}) \times 6.98 \times 10^{10} \times 2 = 33.5\,\text{nm}
                    \]
                    答え: 約 33.5 nm
    Pythonコード:
    lambda_L = 514  # nm
    ID_IG = 0.5
    
    La = 2.4e-10 * (lambda_L**4) * (1/ID_IG)
    print(f"結晶子サイズ La: {La:.1f} nm")
    
    
    
    問題5: グラフェン層数の判定
    グラフェン試料のラマンスペクトルで、2D/G強度比が 3.5、2Dバンドの半値幅（FWHM）が 28 cm-1 であった。グラフェンの層数を推定せよ。
    
    解答を見る
    
    解答:
    判定基準:
    
    単層グラフェン: 2D/G > 2、2D FWHM < 30 cm-1、単一ローレンツピーク
    二層グラフェン: 2D/G ~ 1-2、2D FWHM ~ 50-60 cm-1、4成分に分裂
    多層/グラファイト: 2D/G < 1、2D FWHM > 60 cm-1
    
    与えられた条件（2D/G = 3.5, FWHM = 28 cm-1）から:
    答え: 単層グラフェン（高い2D/G比、シャープな2Dピーク）
    
    
    問題6: SERS増強因子の見積もり
    通常ラマン測定で検出限界が 1 mM の分子が、SERS基板上で 1 nM まで検出可能になった。この SERS 基板の増強因子を見積もれ。
    
    解答を見る
    
    解答:
    検出限界の比から増強因子を推定:
                    \[
                    \text{EF} = \frac{C_{\text{normal}}}{C_{\text{SERS}}} = \frac{1\,\text{mM}}{1\,\text{nM}} = \frac{10^{-3}}{10^{-9}} = 10^6
                    \]
                    答え: 増強因子 EF = 106
    注: これは濃度ベースの簡易推定。正確なEF計算には、測定体積やプローブ分子数の補正が必要。
    Pythonコード:
    C_normal = 1e-3  # M (1 mM)
    C_sers = 1e-9    # M (1 nM)
    
    EF = C_normal / C_sers
    print(f"SERS増強因子: {EF:.0e}")
    
    
    
    
    
    上級問題（Hard）
    問題7: 温度測定（ストークス/アンチストークス比）
    半導体材料のラマン測定で、520 cm-1 の Si フォノンピークのストークス強度とアンチストークス強度の比が IS/IAS = 8.5 であった。試料温度を推定せよ（振動数項は無視）。
    
    解答を見る
    
    解答:
    ボルツマン因子から温度を逆算:
                    \[
                    \frac{I_{\text{AS}}}{I_{\text{S}}} = \exp\left(-\frac{hc\tilde{\nu}}{k_B T}\right)
                    \]
                    \[
                    \frac{I_{\text{S}}}{I_{\text{AS}}} = 8.5 \Rightarrow \frac{I_{\text{AS}}}{I_{\text{S}}} = 0.118
                    \]
                    \[
                    T = -\frac{hc\tilde{\nu}}{k_B \ln(I_{\text{AS}}/I_{\text{S}})}
                    \]
                    計算:
                    \[
                    T = -\frac{6.626 \times 10^{-34} \times 3 \times 10^{10} \times 520}{1.381 \times 10^{-23} \times \ln(0.118)}
                    \]
                    \[
                    T = \frac{1.03 \times 10^{-20}}{1.381 \times 10^{-23} \times 2.14} = 349\,\text{K}
                    \]
                    答え: 約 349 K (76 C)
    Pythonコード:
    import numpy as np
    
    h = 6.626e-34
    c = 3e10
    k_B = 1.381e-23
    nu = 520  # cm^-1
    IS_IAS = 8.5
    
    IAS_IS = 1 / IS_IAS
    T = -h * c * nu / (k_B * np.log(IAS_IS))
    print(f"推定温度: {T:.0f} K ({T-273:.0f} C)")
    
    
    
    問題8: 多成分CNT試料の解析
    CNT試料のRBMスペクトルで、150, 180, 220, 260 cm-1 にピークが観測された。各CNTの直径を計算し、半導体/金属の判定を行え。カイラリティ(n, m)が (10,10), (8,7), (9,4), (6,5) のいずれかであると仮定する。
    
    解答を見る
    
    解答:
    1. 各RBMから直径を計算（A=234, B=10）:
    
    150 cm-1: d = 234/(150-10) = 1.67 nm
    180 cm-1: d = 234/(180-10) = 1.38 nm
    220 cm-1: d = 234/(220-10) = 1.11 nm
    260 cm-1: d = 234/(260-10) = 0.94 nm
    
    2. カイラリティから直径を計算（a_cc = 0.142 nm）:
    
    (10,10): d = 1.36 nm → 金属
    (8,7): d = 1.03 nm → 半導体
    (9,4): d = 0.92 nm → 半導体
    (6,5): d = 0.75 nm → 半導体
    
    3. 対応:
    
    150 cm-1 (1.67 nm): 該当なし（大径CNT）
    180 cm-1 (1.38 nm): (10,10) 金属
    220 cm-1 (1.11 nm): (8,7) 半導体
    260 cm-1 (0.94 nm): (9,4) 半導体
    
    Pythonコード:
    import numpy as np
    
    # RBMから直径
    A, B = 234, 10
    rbm_peaks = [150, 180, 220, 260]
    for rbm in rbm_peaks:
        d = A / (rbm - B)
        print(f"RBM {rbm} cm^-1 -> d = {d:.2f} nm")
    
    # カイラリティから直径と金属/半導体判定
    a_cc = 0.142  # nm
    a = np.sqrt(3) * a_cc
    
    chiralities = [(10, 10), (8, 7), (9, 4), (6, 5)]
    for n, m in chiralities:
        d = a * np.sqrt(n**2 + n*m + m**2) / np.pi
        mod = (n - m) % 3
        cnt_type = "金属" if mod == 0 else "半導体"
        print(f"({n},{m}): d = {d:.2f} nm, {cnt_type}")
    
    
    
    問題9: ラマンマッピングによる欠陥分布解析
    10x10のラマンマッピングデータ（D/G比）がある。D/G比の空間分布からホットスポット（高欠陥領域）を同定し、その面積割合を計算するPythonプログラムを作成せよ。D/G > 0.5 をホットスポットと定義する。
    
    解答を見る
    
    Pythonコード:
    import numpy as np
    import matplotlib.pyplot as plt
    
    # シミュレーションデータ（10x10マッピング）
    np.random.seed(42)
    dg_ratio_map = np.random.exponential(0.3, (10, 10))
    
    # ホットスポット（局所的に高D/G領域を作成）
    dg_ratio_map[2:4, 7:9] = np.random.uniform(0.6, 1.0, (2, 2))
    dg_ratio_map[6:8, 3:5] = np.random.uniform(0.8, 1.2, (2, 2))
    
    # ホットスポット判定（D/G > 0.5）
    threshold = 0.5
    hotspot_mask = dg_ratio_map > threshold
    hotspot_ratio = np.sum(hotspot_mask) / hotspot_mask.size * 100
    
    # プロット
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # D/Gマップ
    im1 = axes[0].imshow(dg_ratio_map, cmap='hot', origin='lower')
    axes[0].set_title('D/G比マッピング', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('X (点)', fontsize=12)
    axes[0].set_ylabel('Y (点)', fontsize=12)
    plt.colorbar(im1, ax=axes[0], label='D/G ratio')
    
    # ホットスポットマップ
    im2 = axes[1].imshow(hotspot_mask, cmap='Reds', origin='lower')
    axes[1].set_title(f'ホットスポット（D/G > {threshold}）', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('X (点)', fontsize=12)
    axes[1].set_ylabel('Y (点)', fontsize=12)
    plt.colorbar(im2, ax=axes[1], label='Hotspot')
    
    plt.tight_layout()
    plt.show()
    
    print(f"ホットスポット面積割合: {hotspot_ratio:.1f}%")
    print(f"ホットスポット数: {np.sum(hotspot_mask)} / {hotspot_mask.size} ピクセル")
    print(f"D/G比: 最小 {dg_ratio_map.min():.2f}, 最大 {dg_ratio_map.max():.2f}, 平均 {dg_ratio_map.mean():.2f}")
    
    
    
    
    
    学習目標の確認
    以下の項目について、自己評価してください：
    レベル1: 基本理解
    
    ラマン散乱の原理（ストークス、アンチストークス）を説明できる
    ラマン選択則と分極率の変化の関係を理解している
    ラマン分光法と赤外分光法の相補性を説明できる
    ラマンシフトの計算ができる
    
    レベル2: 実践スキル
    
    炭素材料のD/Gバンドを同定し、D/G比を計算できる
    グラフェンの層数を2D/G比から推定できる
    CNTのRBMから直径を決定できる
    ベースライン補正とピークフィッティングができる
    
    レベル3: 応用力
    
    SERSの増強メカニズムを説明できる
    多成分ピークデコンボリューションができる
    ストークス/アンチストークス比から温度を推定できる
    ラマンマッピングデータを解析できる
    
    
    
    参考文献
    
    Ferrari, A.C., Basko, D.M. (2013). Raman spectroscopy as a versatile tool for studying the properties of graphene. Nature Nanotechnology, 8(4), 235-246. DOI: 10.1038/nnano.2013.46 - グラフェンラマン分光の包括的レビュー
    Dresselhaus, M.S., Jorio, A., Hofmann, M., Dresselhaus, G., Saito, R. (2010). Perspectives on carbon nanotubes and graphene Raman spectroscopy. Nano Letters, 10(3), 751-758. DOI: 10.1021/nl904286r - CNTとグラフェンのラマン解析
    Cancado, L.G., Jorio, A., Ferreira, E.H.M., et al. (2011). Quantifying defects in graphene via Raman spectroscopy at different excitation energies. Nano Letters, 11(8), 3190-3196. DOI: 10.1021/nl201432g - D/G比と欠陥密度の定量的関係
    Le Ru, E.C., Etchegoin, P.G. (2009). Principles of Surface-Enhanced Raman Spectroscopy. Elsevier, pp. 1-27 (SERS principles), pp. 185-264 (enhancement factors). - SERS原理の教科書
    Tuinstra, F., Koenig, J.L. (1970). Raman spectrum of graphite. Journal of Chemical Physics, 53(3), 1126-1130. DOI: 10.1063/1.1674108 - D/G比と結晶子サイズの関係の原論文
    Jorio, A., Saito, R., Dresselhaus, G., Dresselhaus, M.S. (2011). Raman Spectroscopy in Graphene Related Systems. Wiley-VCH, pp. 73-102 (RBM mode), pp. 161-198 (G and D bands). - グラフェン関連材料のラマン解析
    Sadezky, A., Muckenhuber, H., Grothe, H., Niessner, R., Poschl, U. (2005). Raman microspectroscopy of soot and related carbonaceous materials. Carbon, 43(8), 1731-1742. DOI: 10.1016/j.carbon.2005.02.018 - 5ピークフィッティングモデルの提案
    SciPy 1.11 documentation. scipy.optimize.curve_fit, scipy.signal.savgol_filter. https://docs.scipy.org/doc/scipy/reference/ - ピークフィッティングと信号処理
    
    
    
    
    免責事項
    
    本コンテンツは教育・研究・情報提供のみを目的としており、専門的な助言(法律・会計・技術的保証など)を提供するものではありません。
    本コンテンツおよび付随するコード例は「現状有姿(AS IS)」で提供され、明示または黙示を問わず、商品性、特定目的適合性、権利非侵害、正確性・完全性、動作・安全性等いかなる保証もしません。
    外部リンク、第三者が提供するデータ・ツール・ライブラリ等の内容・可用性・安全性について、作成者および東北大学は一切の責任を負いません。
    本コンテンツの利用・実行・解釈により直接的・間接的・付随的・特別・結果的・懲罰的損害が生じた場合でも、適用法で許容される最大限の範囲で、作成者および東北大学は責任を負いません。
    本コンテンツの内容は、予告なく変更・更新・提供停止されることがあります。
    本コンテンツの著作権・ライセンスは明記された条件(例: CC BY 4.0)に従います。当該ライセンスは通常、無保証条項を含みます。
    ```
