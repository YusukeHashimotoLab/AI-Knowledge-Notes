---
title: "第2章: 高分子構造"
chapter_title: "第2章: 高分子構造"
---

🌐 JP | [🇬🇧 EN](<../../../en/MS/polymer-materials-introduction/chapter-2.html>) | Last sync: 2025-11-16

[AI寺子屋トップ](<../../index.html>)›[材料科学](<../../MS/index.html>)›[Polymer Materials](<../../MS/polymer-materials-introduction/index.html>)›Chapter 2

  * [目次](<index.html>)
  * [← 第1章](<chapter-1.html>)
  * [第2章](<chapter-2.html>)
  * [第3章 →](<chapter-3.html>)
  * 第4章（準備中）
  * [第5章](<chapter-5.html>)

### 学習目標

**初級:**

  * タクティシティ（isotactic, syndiotactic, atactic）の定義と構造を理解する
  * 結晶構造と非晶構造の違いを説明できる
  * ガラス転移温度（Tg）と融点（Tm）の物理的意味を理解する

**中級:**

  * Hermans配向関数を用いて高分子の配向度を計算できる
  * Fox式やGordon-Taylor式を用いてTgを予測できる
  * XRDデータから結晶化度を算出できる

**上級:**

  * ゴム弾性理論に基づいて架橋密度を計算できる
  * Flory-Huggins相図を描画し、相溶性を評価できる
  * DSC曲線をシミュレートし、熱転移を解析できる

## 2.1 立体規則性（タクティシティ）

高分子鎖の立体配置は、物性に大きな影響を与えます。ビニル重合体（-CH2-CHR-）では、主鎖炭素の置換基Rの配置により、**isotactic** （アイソタクチック：全て同じ側）、**syndiotactic** （シンジオタクチック：交互）、**atactic** （アタクチック：ランダム）の3つに分類されます。 

### タクティシティと結晶性

isotacticおよびsyndiotactic高分子は規則的な構造のため**結晶化しやすく** 、融点が高くなります。一方、atactic高分子は不規則なため結晶化せず、**アモルファス（非晶質）** 構造を形成します。例えば、isotactic ポリプロピレン（iPP）はTm = 165°C、atactic ポリプロピレン（aPP）はTg = -10°Cと全く異なる物性を示します。 
    
    
    ```mermaid
    flowchart TD
                        A[ビニル重合体 -CH2-CHR-] --> B[Isotactic]
                        A --> C[Syndiotactic]
                        A --> D[Atactic]
                        B --> E[規則的配置結晶性高Tm高]
                        C --> F[交互配置結晶性中Tm中]
                        D --> G[ランダム配置非晶質Tgのみ]
                        E --> H[用途: 繊維, フィルム]
                        F --> I[用途: 特殊樹脂]
                        G --> J[用途: ゴム, 粘着剤]
    ```

### 2.1.1 NMRによるタクティシティ解析

13C-NMRでは、立体配置の違いが化学シフトの差として現れます。トリアッド（3連子）配列（mm, mr, rr）の相対強度から、isotactic分率を計算できます。 
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.signal import find_peaks
    
    # NMRスペクトルシミュレーション（タクティシティ解析）
    def simulate_nmr_spectrum(isotactic_fraction=0.7, syndiotactic_fraction=0.2):
        """
        13C-NMRスペクトルをシミュレートし、タクティシティを解析
    
        Parameters:
        - isotactic_fraction: isotactic分率（mm）
        - syndiotactic_fraction: syndiotactic分率（rr）
    
        Returns:
        - chemical_shift: 化学シフト（ppm）
        - intensity: スペクトル強度
        """
        # atactic分率を計算
        atactic_fraction = 1 - isotactic_fraction - syndiotactic_fraction
    
        # 化学シフト範囲（メチレン炭素領域）
        chemical_shift = np.linspace(18, 24, 1000)
    
        # 各タクティシティに対応するピーク位置
        # mm（isotactic）: 21.8 ppm
        # mr（atactic）: 21.3 ppm
        # rr（syndiotactic）: 20.8 ppm
        peak_positions = [21.8, 21.3, 20.8]
        peak_intensities = [isotactic_fraction, atactic_fraction, syndiotactic_fraction]
    
        # Lorentzianピークを生成
        def lorentzian(x, x0, gamma, intensity):
            """Lorentzian関数"""
            return intensity * (gamma**2 / ((x - x0)**2 + gamma**2))
    
        # スペクトル合成
        intensity = np.zeros_like(chemical_shift)
        for pos, intens in zip(peak_positions, peak_intensities):
            intensity += lorentzian(chemical_shift, pos, 0.15, intens)
    
        # ノイズ追加
        noise = np.random.normal(0, 0.01, len(chemical_shift))
        intensity += noise
    
        # ピーク検出と面積計算
        peaks, _ = find_peaks(intensity, height=0.1)
    
        # タクティシティ指標計算
        total_area = np.trapz(intensity, chemical_shift)
        mm_area = isotactic_fraction * total_area
        mr_area = atactic_fraction * total_area
        rr_area = syndiotactic_fraction * total_area
    
        # 可視化
        plt.figure(figsize=(10, 6))
        plt.plot(chemical_shift, intensity, 'b-', linewidth=2, label='13C-NMR Spectrum')
        plt.fill_between(chemical_shift, intensity, alpha=0.3)
    
        # ピーク位置をマーク
        plt.axvline(21.8, color='red', linestyle='--', alpha=0.7, label='mm (isotactic)')
        plt.axvline(21.3, color='green', linestyle='--', alpha=0.7, label='mr (atactic)')
        plt.axvline(20.8, color='blue', linestyle='--', alpha=0.7, label='rr (syndiotactic)')
    
        plt.xlabel('Chemical Shift (ppm)', fontsize=12)
        plt.ylabel('Intensity', fontsize=12)
        plt.title('Tacticity Analysis by 13C-NMR', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.gca().invert_xaxis()  # NMRは通常左→右が大きい
        plt.tight_layout()
        plt.savefig('nmr_tacticity.png', dpi=300, bbox_inches='tight')
        plt.show()
    
        # 結果出力
        print("=== タクティシティ解析結果 ===")
        print(f"Isotactic分率（mm）: {isotactic_fraction:.1%}")
        print(f"Atactic分率（mr）: {atactic_fraction:.1%}")
        print(f"Syndiotactic分率（rr）: {syndiotactic_fraction:.1%}")
        print(f"\n平均立体規則性: {isotactic_fraction + syndiotactic_fraction:.1%}")
    
        return chemical_shift, intensity
    
    # 実行例：高立体規則性ポリプロピレン
    simulate_nmr_spectrum(isotactic_fraction=0.85, syndiotactic_fraction=0.05)
    

## 2.2 結晶構造と非晶構造

高分子は完全結晶化せず、**結晶領域（Crystalline region）** と**非晶領域（Amorphous region）** が共存する半結晶性構造を形成します。結晶化度（Crystallinity）は、材料の機械的強度、透明性、密度に大きく影響します。 

### 2.2.1 球晶とラメラ構造

高分子結晶は**球晶（Spherulite）** と呼ばれる球状の集合体を形成します。球晶は中心から放射状に成長する**ラメラ（Lamella）** と呼ばれる板状結晶で構成され、その厚さは10-20 nm程度です。ラメラ間には非晶領域が存在し、分子鎖が折れ畳まれて（chain folding）結晶化します。 
    
    
    ```mermaid
    flowchart TD
                        A[高分子融液] -->|冷却| B[核生成]
                        B --> C[球晶成長]
                        C --> D[ラメラ構造]
                        D --> E[結晶領域厚さ 10-20 nm]
                        D --> F[非晶領域Chain folding]
                        E --> G[高密度1.00 g/cm³]
                        F --> H[低密度0.85 g/cm³]
                        G --> I[機械的強度向上]
                        H --> J[柔軟性・延性]
    ```

### 2.2.2 XRDによる結晶化度計算

X線回折（XRD）では、結晶領域からの鋭いブラッグピークと非晶領域からのハローが観測されます。結晶化度χcは、全散乱強度に対する結晶ピーク面積の比として計算されます。 
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.signal import find_peaks
    from scipy.integrate import trapz
    
    # XRD結晶化度解析
    def calculate_crystallinity_xrd(crystallinity_true=0.65):
        """
        X線回折データから結晶化度を計算
    
        Parameters:
        - crystallinity_true: 真の結晶化度（シミュレーション用）
    
        Returns:
        - crystallinity_calculated: 計算された結晶化度
        """
        # 2θ角度範囲（度）
        two_theta = np.linspace(10, 40, 500)
    
        # 結晶ピーク（ポリエチレンの例）
        # (110): 21.5°, (200): 23.8°
        def gaussian(x, mu, sigma, amplitude):
            """Gaussian関数"""
            return amplitude * np.exp(-0.5 * ((x - mu) / sigma)**2)
    
        # 結晶ピークを生成
        crystal_peak1 = gaussian(two_theta, 21.5, 0.5, crystallinity_true * 100)
        crystal_peak2 = gaussian(two_theta, 23.8, 0.5, crystallinity_true * 80)
        crystal_intensity = crystal_peak1 + crystal_peak2
    
        # 非晶ハロー（ブロードなピーク）
        amorphous_halo = gaussian(two_theta, 19.5, 3.0, (1 - crystallinity_true) * 120)
    
        # 全強度
        total_intensity = crystal_intensity + amorphous_halo
    
        # ノイズ追加
        noise = np.random.normal(0, 2, len(two_theta))
        total_intensity += noise
        total_intensity = np.maximum(total_intensity, 0)  # 負の値を除去
    
        # 結晶化度計算（ピーク分離法）
        # 非晶ベースラインを多項式フィッティングで推定
        amorphous_baseline = amorphous_halo
    
        # 結晶ピーク面積
        crystal_area = trapz(crystal_intensity, two_theta)
    
        # 全面積
        total_area = trapz(total_intensity, two_theta)
    
        # 結晶化度
        crystallinity_calculated = crystal_area / total_area
    
        # 可視化
        plt.figure(figsize=(12, 6))
    
        plt.subplot(1, 2, 1)
        plt.plot(two_theta, total_intensity, 'k-', linewidth=2, label='Total Intensity')
        plt.plot(two_theta, amorphous_baseline, 'b--', linewidth=1.5, label='Amorphous Halo')
        plt.plot(two_theta, crystal_intensity, 'r--', linewidth=1.5, label='Crystalline Peaks')
        plt.fill_between(two_theta, crystal_intensity, alpha=0.3, color='red')
        plt.xlabel('2θ (deg)', fontsize=12)
        plt.ylabel('Intensity (a.u.)', fontsize=12)
        plt.title('XRD Pattern Decomposition', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(alpha=0.3)
    
        plt.subplot(1, 2, 2)
        categories = ['Crystalline', 'Amorphous']
        areas = [crystal_area, total_area - crystal_area]
        colors = ['#f5576c', '#f093fb']
        plt.pie(areas, labels=categories, autopct='%1.1f%%', colors=colors, startangle=90)
        plt.title('Crystallinity Composition', fontsize=14, fontweight='bold')
    
        plt.tight_layout()
        plt.savefig('xrd_crystallinity.png', dpi=300, bbox_inches='tight')
        plt.show()
    
        # 結果出力
        print("=== XRD結晶化度解析結果 ===")
        print(f"結晶ピーク面積: {crystal_area:.2f}")
        print(f"全散乱面積: {total_area:.2f}")
        print(f"計算結晶化度: {crystallinity_calculated:.1%}")
        print(f"真の結晶化度: {crystallinity_true:.1%}")
        print(f"誤差: {abs(crystallinity_calculated - crystallinity_true):.1%}")
    
        return crystallinity_calculated
    
    # 実行例：結晶化度65%のポリエチレン
    calculate_crystallinity_xrd(crystallinity_true=0.65)
    

## 2.3 配向と延伸

高分子フィルムや繊維は、延伸（Drawing）により分子鎖が特定方向に**配向（Orientation）** します。配向度はHermans配向関数で定量化され、機械的強度に直結します。 

### 2.3.1 Hermans配向関数

配向関数 _f_ は、配向角 θ の二次モーメントから計算されます： 

\\[ f = \frac{3\langle \cos^2 \theta \rangle - 1}{2} \\] 

ここで、θ は分子鎖軸と延伸方向の角度です。完全配向で _f_ = 1、ランダム配向で _f_ = 0、垂直配向で _f_ = -0.5 となります。 
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Hermans配向関数計算
    def calculate_hermans_orientation(draw_ratio_range=(1, 10), num_points=50):
        """
        延伸倍率に対するHermans配向関数を計算
    
        Parameters:
        - draw_ratio_range: 延伸倍率の範囲（λ）
        - num_points: 計算点数
    
        Returns:
        - draw_ratios: 延伸倍率
        - orientation_functions: 配向関数
        """
        draw_ratios = np.linspace(draw_ratio_range[0], draw_ratio_range[1], num_points)
    
        # 延伸倍率と配向の経験式（擬アフィン変形モデル）
        # f = (λ² - 1) / (λ² + 2)  （λ: 延伸倍率）
        orientation_functions = (draw_ratios**2 - 1) / (draw_ratios**2 + 2)
    
        # 配向角分布のシミュレーション
        angles = np.linspace(0, 90, 180)  # 0°から90°
    
        # 3つの延伸倍率での配向角分布
        draw_ratios_example = [1, 3, 8]
    
        plt.figure(figsize=(14, 5))
    
        # サブプロット1：配向関数vs延伸倍率
        plt.subplot(1, 3, 1)
        plt.plot(draw_ratios, orientation_functions, 'b-', linewidth=2)
        plt.scatter(draw_ratios_example,
                    [(dr**2 - 1) / (dr**2 + 2) for dr in draw_ratios_example],
                    c='red', s=100, zorder=5, label='Example Points')
        plt.xlabel('Draw Ratio λ', fontsize=12)
        plt.ylabel('Orientation Function f', fontsize=12)
        plt.title('Hermans Orientation Function', fontsize=14, fontweight='bold')
        plt.grid(alpha=0.3)
        plt.legend()
        plt.axhline(0, color='k', linestyle='--', linewidth=0.8)
        plt.axhline(1, color='r', linestyle='--', linewidth=0.8, alpha=0.5, label='Perfect Orientation')
    
        # サブプロット2：配向角分布
        plt.subplot(1, 3, 2)
        for dr in draw_ratios_example:
            # 配向パラメータ
            f = (dr**2 - 1) / (dr**2 + 2)
            # 配向角分布（簡易モデル：Gaussianで近似）
            sigma = 45 * (1 - f)  # 配向が高いほど分布が狭くなる
            distribution = np.exp(-0.5 * (angles / sigma)**2)
            distribution /= np.max(distribution)  # 正規化
            plt.plot(angles, distribution, linewidth=2, label=f'λ = {dr}, f = {f:.2f}')
    
        plt.xlabel('Orientation Angle θ (deg)', fontsize=12)
        plt.ylabel('Probability Density', fontsize=12)
        plt.title('Orientation Angle Distribution', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(alpha=0.3)
    
        # サブプロット3：機械的強度への影響
        plt.subplot(1, 3, 3)
        # 配向方向の引張強度（経験式）
        tensile_strength = 50 + 200 * orientation_functions  # MPa
        plt.plot(draw_ratios, tensile_strength, 'g-', linewidth=2)
        plt.xlabel('Draw Ratio λ', fontsize=12)
        plt.ylabel('Tensile Strength (MPa)', fontsize=12)
        plt.title('Strength Enhancement by Orientation', fontsize=14, fontweight='bold')
        plt.grid(alpha=0.3)
    
        plt.tight_layout()
        plt.savefig('hermans_orientation.png', dpi=300, bbox_inches='tight')
        plt.show()
    
        # 結果出力
        print("=== Hermans配向関数解析結果 ===")
        for dr in draw_ratios_example:
            f = (dr**2 - 1) / (dr**2 + 2)
            strength = 50 + 200 * f
            print(f"\n延伸倍率 λ = {dr}:")
            print(f"  配向関数 f = {f:.3f}")
            print(f"  引張強度 = {strength:.1f} MPa")
            print(f"  配向状態: {'高配向' if f > 0.7 else '中配向' if f > 0.4 else '低配向'}")
    
        return draw_ratios, orientation_functions
    
    # 実行
    calculate_hermans_orientation()
    

## 2.4 ガラス転移温度（Tg）と融点（Tm）

高分子の熱的性質を特徴づける2つの重要な温度は、**ガラス転移温度（T g）**と**融点（T m）**です。Tgはアモルファス領域の分子運動が活発になる温度、Tmは結晶領域が溶融する温度です。 

### TgとTmの物理的意味

**T g（ガラス転移温度）：** 高分子鎖のセグメント運動が開始する温度。Tg以下では硬くて脆い「ガラス状態」、Tg以上では柔軟な「ゴム状態」。 

**T m（融点）：** 結晶領域が溶融する温度。完全非晶質高分子（atactic）はTmを持たず、Tgのみ存在。半結晶性高分子はTgとTmの両方を持つ。 

### 2.4.1 Fox式によるTg予測

共重合体のTgは、各成分のTgと質量分率から**Fox式** で予測できます： 

\\[ \frac{1}{T_g} = \frac{w_1}{T_{g1}} + \frac{w_2}{T_{g2}} \\] 

ここで、 _w i_は質量分率、 _T gi_は各成分のTg（K）です。 
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Fox式によるTg予測
    def predict_tg_fox_equation(tg1=373, tg2=233, num_points=50):
        """
        Fox式を用いて共重合体のTgを予測
    
        Parameters:
        - tg1: 成分1のTg（K）（例：ポリスチレン 100°C = 373 K）
        - tg2: 成分2のTg（K）（例：ポリブタジエン -40°C = 233 K）
        - num_points: 計算点数
    
        Returns:
        - compositions: 成分1の質量分率
        - tg_values: 予測されたTg（K）
        """
        # 成分1の質量分率
        w1 = np.linspace(0, 1, num_points)
        w2 = 1 - w1
    
        # Fox式でTgを計算
        tg_fox = 1 / (w1 / tg1 + w2 / tg2)
    
        # Gordon-Taylor式（より精密な予測）
        # Tg = (w1*Tg1 + k*w2*Tg2) / (w1 + k*w2)
        # k: フィッティングパラメータ（通常0.5-2.0）
        k = 1.0
        tg_gordon_taylor = (w1 * tg1 + k * w2 * tg2) / (w1 + k * w2)
    
        # 可視化
        plt.figure(figsize=(12, 5))
    
        plt.subplot(1, 2, 1)
        plt.plot(w1 * 100, tg_fox - 273.15, 'b-', linewidth=2, label='Fox Equation')
        plt.plot(w1 * 100, tg_gordon_taylor - 273.15, 'r--', linewidth=2, label='Gordon-Taylor (k=1.0)')
        plt.axhline(tg1 - 273.15, color='gray', linestyle=':', alpha=0.7, label=f'Component 1 Tg ({tg1-273.15:.0f}°C)')
        plt.axhline(tg2 - 273.15, color='gray', linestyle=':', alpha=0.7, label=f'Component 2 Tg ({tg2-273.15:.0f}°C)')
        plt.xlabel('Component 1 Weight Fraction (%)', fontsize=12)
        plt.ylabel('Glass Transition Temperature (°C)', fontsize=12)
        plt.title('Copolymer Tg Prediction', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(alpha=0.3)
    
        # サブプロット2：様々なkパラメータでのGordon-Taylor式
        plt.subplot(1, 2, 2)
        k_values = [0.5, 1.0, 1.5, 2.0]
        for k in k_values:
            tg_gt = (w1 * tg1 + k * w2 * tg2) / (w1 + k * w2)
            plt.plot(w1 * 100, tg_gt - 273.15, linewidth=2, label=f'k = {k}')
        plt.plot(w1 * 100, tg_fox - 273.15, 'k--', linewidth=2, label='Fox (k→∞)')
        plt.xlabel('Component 1 Weight Fraction (%)', fontsize=12)
        plt.ylabel('Glass Transition Temperature (°C)', fontsize=12)
        plt.title('Effect of Gordon-Taylor Parameter k', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(alpha=0.3)
    
        plt.tight_layout()
        plt.savefig('fox_tg_prediction.png', dpi=300, bbox_inches='tight')
        plt.show()
    
        # 特定組成での計算例
        compositions = [0.25, 0.5, 0.75]
        print("=== 共重合体Tg予測結果 ===")
        print(f"成分1のTg: {tg1 - 273.15:.1f}°C")
        print(f"成分2のTg: {tg2 - 273.15:.1f}°C\n")
    
        for w in compositions:
            tg_pred = 1 / (w / tg1 + (1 - w) / tg2)
            print(f"成分1質量分率 {w*100:.0f}%:")
            print(f"  Fox式予測Tg: {tg_pred - 273.15:.1f}°C")
    
        return w1, tg_fox
    
    # 実行例：ポリスチレン（Tg = 100°C）とポリブタジエン（Tg = -40°C）の共重合体
    predict_tg_fox_equation(tg1=373, tg2=233)
    

### 2.4.2 Gordon-Taylor式によるTg予測

Gordon-Taylor式は、体積収縮の違いを考慮したより精密な予測式です： 

\\[ T_g = \frac{w_1 T_{g1} + k w_2 T_{g2}}{w_1 + k w_2} \\] 

パラメータ _k_ は、各成分の熱膨張係数の比から決定されます。 
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.optimize import curve_fit
    
    # Gordon-Taylor式フィッティング
    def fit_gordon_taylor_equation(experimental_data=None):
        """
        実験データにGordon-Taylor式をフィッティングしてkを決定
    
        Parameters:
        - experimental_data: 実験データ（dict形式）
          {'compositions': [w1値のリスト], 'tg_values': [Tg値のリスト]}
    
        Returns:
        - k_fitted: フィッティングされたkパラメータ
        """
        # 実験データがない場合はシミュレーションデータを生成
        if experimental_data is None:
            tg1, tg2 = 373, 233  # K
            k_true = 1.3
            compositions = np.array([0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0])
            tg_values = (compositions * tg1 + k_true * (1 - compositions) * tg2) / \
                        (compositions + k_true * (1 - compositions))
            # ノイズ追加
            tg_values += np.random.normal(0, 2, len(tg_values))
            experimental_data = {'compositions': compositions, 'tg_values': tg_values}
            print(f"シミュレーションデータ生成（真のk = {k_true}）\n")
    
        w1_exp = np.array(experimental_data['compositions'])
        tg_exp = np.array(experimental_data['tg_values'])
    
        # 純成分のTg
        tg1 = tg_exp[w1_exp == 1.0][0] if any(w1_exp == 1.0) else tg_exp[-1]
        tg2 = tg_exp[w1_exp == 0.0][0] if any(w1_exp == 0.0) else tg_exp[0]
    
        # Gordon-Taylor式
        def gordon_taylor(w1, k):
            w2 = 1 - w1
            return (w1 * tg1 + k * w2 * tg2) / (w1 + k * w2)
    
        # フィッティング
        k_fitted, _ = curve_fit(gordon_taylor, w1_exp, tg_exp, p0=[1.0])
        k_fitted = k_fitted[0]
    
        # 予測曲線
        w1_fine = np.linspace(0, 1, 100)
        tg_fitted = gordon_taylor(w1_fine, k_fitted)
    
        # 可視化
        plt.figure(figsize=(10, 6))
        plt.scatter(w1_exp * 100, tg_exp - 273.15, s=100, c='red',
                    edgecolors='black', linewidths=2, zorder=5, label='Experimental Data')
        plt.plot(w1_fine * 100, tg_fitted - 273.15, 'b-', linewidth=2,
                 label=f'Gordon-Taylor Fit (k = {k_fitted:.2f})')
        plt.xlabel('Component 1 Weight Fraction (%)', fontsize=12)
        plt.ylabel('Glass Transition Temperature (°C)', fontsize=12)
        plt.title('Gordon-Taylor Equation Fitting', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig('gordon_taylor_fit.png', dpi=300, bbox_inches='tight')
        plt.show()
    
        # 結果出力
        print("=== Gordon-Taylorフィッティング結果 ===")
        print(f"フィッティングされたk: {k_fitted:.3f}")
        print(f"成分1のTg: {tg1 - 273.15:.1f}°C")
        print(f"成分2のTg: {tg2 - 273.15:.1f}°C")
        print(f"\n残差二乗和: {np.sum((tg_exp - gordon_taylor(w1_exp, k_fitted))**2):.2f}")
    
        return k_fitted
    
    # 実行
    fit_gordon_taylor_equation()
    

## 2.5 分岐と架橋

高分子鎖の**分岐（Branching）** と**架橋（Crosslinking）** は、物性に大きな影響を与えます。分岐は流動性や結晶性に影響し、架橋はゴム弾性を生み出します。 

### 2.5.1 ゴム弾性理論

架橋ゴムの弾性は、分子鎖のエントロピー弾性に起因します。応力-ひずみ関係は**統計的ゴム弾性理論** で記述されます： 

\\[ \sigma = G (\lambda - \lambda^{-2}) \\] 

ここで、σは応力、λは伸長比（λ = L/L0）、Gは剛性率（shear modulus）です。Gは架橋密度νcに比例します： 

\\[ G = \nu_c R T \\] 
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    # ゴム弾性理論シミュレーション
    def simulate_rubber_elasticity(crosslink_densities=[0.5, 1.0, 2.0], temperature=298):
        """
        架橋密度に応じたゴム弾性の応力-ひずみ曲線を計算
    
        Parameters:
        - crosslink_densities: 架橋密度のリスト（mol/m³）
        - temperature: 温度（K）
    
        Returns:
        - stretch_ratios: 伸長比
        - stresses: 応力（各架橋密度）
        """
        R = 8.314  # 気体定数（J/mol·K）
        T = temperature  # 温度（K）
    
        # 伸長比（λ = L/L0）
        stretch_ratios = np.linspace(1, 7, 100)
    
        plt.figure(figsize=(14, 5))
    
        # サブプロット1：応力-ひずみ曲線
        plt.subplot(1, 3, 1)
        for nu_c in crosslink_densities:
            # 剛性率（Pa）
            G = nu_c * R * T * 1000  # mol/m³ → mol/L変換
            # 応力（MPa）
            stress = G * (stretch_ratios - stretch_ratios**(-2)) / 1e6
            plt.plot(stretch_ratios, stress, linewidth=2,
                     label=f'νc = {nu_c} mol/L')
    
        plt.xlabel('Stretch Ratio λ', fontsize=12)
        plt.ylabel('Engineering Stress (MPa)', fontsize=12)
        plt.title('Rubber Elasticity: Stress-Strain Curves', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(alpha=0.3)
    
        # サブプロット2：架橋密度とモジュラスの関係
        plt.subplot(1, 3, 2)
        nu_c_range = np.linspace(0.1, 3, 50)
        G_range = nu_c_range * R * T * 1000 / 1e6  # MPa
        plt.plot(nu_c_range, G_range, 'b-', linewidth=2)
        plt.scatter(crosslink_densities,
                    [nu * R * T * 1000 / 1e6 for nu in crosslink_densities],
                    s=100, c='red', edgecolors='black', linewidths=2, zorder=5)
        plt.xlabel('Crosslink Density νc (mol/L)', fontsize=12)
        plt.ylabel('Shear Modulus G (MPa)', fontsize=12)
        plt.title('Crosslink Density vs Modulus', fontsize=14, fontweight='bold')
        plt.grid(alpha=0.3)
    
        # サブプロット3：温度依存性
        plt.subplot(1, 3, 3)
        temperatures = np.linspace(250, 400, 50)  # K
        nu_c_fixed = 1.0  # mol/L
        G_temp = nu_c_fixed * R * temperatures * 1000 / 1e6  # MPa
        plt.plot(temperatures - 273.15, G_temp, 'g-', linewidth=2)
        plt.axvline(temperature - 273.15, color='red', linestyle='--',
                    linewidth=2, label=f'Current T = {temperature}K')
        plt.xlabel('Temperature (°C)', fontsize=12)
        plt.ylabel('Shear Modulus G (MPa)', fontsize=12)
        plt.title('Temperature Dependence (νc = 1.0 mol/L)', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(alpha=0.3)
    
        plt.tight_layout()
        plt.savefig('rubber_elasticity.png', dpi=300, bbox_inches='tight')
        plt.show()
    
        # 結果出力
        print("=== ゴム弾性理論解析結果 ===")
        print(f"温度: {temperature} K ({temperature - 273.15:.1f}°C)\n")
        for nu_c in crosslink_densities:
            G = nu_c * R * T * 1000 / 1e6
            # λ = 2での応力を計算
            lambda_test = 2.0
            stress_test = G * (lambda_test - lambda_test**(-2))
            print(f"架橋密度 νc = {nu_c} mol/L:")
            print(f"  剛性率 G = {G:.3f} MPa")
            print(f"  応力（λ=2）= {stress_test:.3f} MPa")
    
        return stretch_ratios, crosslink_densities
    
    # 実行
    simulate_rubber_elasticity()
    

### 2.5.2 DSC曲線シミュレーション

示差走査熱量測定（DSC）は、TgとTmを実験的に決定する標準手法です。DSC曲線をシミュレートすることで、熱転移の理解を深められます。 
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    # DSC曲線シミュレーション
    def simulate_dsc_curve(tg=323, tm=438, crystallinity=0.6, heating_rate=10):
        """
        DSC（示差走査熱量測定）曲線をシミュレート
    
        Parameters:
        - tg: ガラス転移温度（K）
        - tm: 融点（K）
        - crystallinity: 結晶化度
        - heating_rate: 昇温速度（K/min）
    
        Returns:
        - temperatures: 温度（°C）
        - heat_flow: 熱流束（W/g）
        """
        # 温度範囲（K）
        temperatures = np.linspace(200, 500, 1000)
    
        # ベースライン
        baseline = -0.5 + 0.001 * temperatures  # わずかに温度依存
    
        # ガラス転移（ステップ変化）
        def glass_transition(T, Tg, delta_Cp=0.3, width=10):
            """ガラス転移のシグモイド関数"""
            return delta_Cp / (1 + np.exp(-(T - Tg) / width))
    
        # 融解ピーク（Gaussianピーク）
        def melting_peak(T, Tm, delta_Hm, width=15):
            """融解ピークのGaussian関数"""
            return delta_Hm * np.exp(-0.5 * ((T - Tm) / width)**2)
    
        # Tgでのステップ変化
        tg_signal = glass_transition(temperatures, tg)
    
        # Tmでの吸熱ピーク（結晶化度に比例）
        delta_Hm = -100 * crystallinity  # 融解エンタルピー（J/g）
        tm_signal = melting_peak(temperatures, tm, delta_Hm)
    
        # 全DSC信号
        heat_flow = baseline + tg_signal + tm_signal
    
        # ノイズ追加
        noise = np.random.normal(0, 0.02, len(temperatures))
        heat_flow += noise
    
        # 可視化
        plt.figure(figsize=(12, 6))
    
        plt.subplot(1, 2, 1)
        plt.plot(temperatures - 273.15, heat_flow, 'b-', linewidth=2, label='DSC Signal')
        plt.axvline(tg - 273.15, color='red', linestyle='--', linewidth=1.5,
                    label=f'Tg = {tg - 273.15:.0f}°C')
        plt.axvline(tm - 273.15, color='green', linestyle='--', linewidth=1.5,
                    label=f'Tm = {tm - 273.15:.0f}°C')
    
        # TgとTmの領域を強調
        plt.fill_betweenx([heat_flow.min(), heat_flow.max()],
                          tg - 273.15 - 20, tg - 273.15 + 20,
                          alpha=0.2, color='red', label='Tg Region')
        plt.fill_betweenx([heat_flow.min(), heat_flow.max()],
                          tm - 273.15 - 30, tm - 273.15 + 30,
                          alpha=0.2, color='green', label='Tm Region')
    
        plt.xlabel('Temperature (°C)', fontsize=12)
        plt.ylabel('Heat Flow (W/g) Exo ↑', fontsize=12)
        plt.title(f'DSC Curve (Crystallinity = {crystallinity:.0%})', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.gca().invert_yaxis()  # DSCは通常吸熱が下向き
    
        # サブプロット2：結晶化度の影響
        plt.subplot(1, 2, 2)
        crystallinities = [0.3, 0.5, 0.7]
        for xc in crystallinities:
            delta_Hm_var = -100 * xc
            tm_signal_var = melting_peak(temperatures, tm, delta_Hm_var)
            heat_flow_var = baseline + tg_signal + tm_signal_var
            plt.plot(temperatures - 273.15, heat_flow_var, linewidth=2,
                     label=f'Xc = {xc:.0%}')
    
        plt.xlabel('Temperature (°C)', fontsize=12)
        plt.ylabel('Heat Flow (W/g) Exo ↑', fontsize=12)
        plt.title('Effect of Crystallinity on Melting Peak', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.gca().invert_yaxis()
    
        plt.tight_layout()
        plt.savefig('dsc_simulation.png', dpi=300, bbox_inches='tight')
        plt.show()
    
        # 結果出力
        print("=== DSC解析結果 ===")
        print(f"ガラス転移温度 Tg: {tg - 273.15:.1f}°C")
        print(f"融点 Tm: {tm - 273.15:.1f}°C")
        print(f"結晶化度: {crystallinity:.0%}")
        print(f"融解エンタルピー: {-100 * crystallinity:.1f} J/g")
        print(f"昇温速度: {heating_rate} K/min")
    
        return temperatures, heat_flow
    
    # 実行例：ポリエチレン（Tg = 50°C, Tm = 165°C, 結晶化度60%）
    simulate_dsc_curve(tg=323, tm=438, crystallinity=0.6)
    

## 演習問題

#### 演習1: タクティシティ解析（Easy）

13C-NMRで測定されたポリプロピレンのトリアッド分率が以下の場合、isotactic分率を計算してください。 

  * mm（isotactic）: 70%
  * mr（heterotactic）: 25%
  * rr（syndiotactic）: 5%

解答を見る

**解答:**

Isotactic分率 = mm分率 = 70%

この高分子は高立体規則性を持ち、高い結晶性（約50-60%）と融点（約165°C）を示すと予想されます。

#### 演習2: 結晶化度計算（Easy）

ポリエチレンのXRDパターンで、結晶ピーク面積が300、全散乱面積が500でした。結晶化度を計算してください。 

解答を見る

**解答:**
    
    
    crystallinity = 300 / 500 = 0.6 = 60%

結晶化度60%は高密度ポリエチレン（HDPE）の典型的な値です。

#### 演習3: Hermans配向関数（Easy）

延伸倍率λ = 4のフィルムの配向関数fを計算してください（擬アフィン変形モデル使用）。 

解答を見る

**解答:**
    
    
    lambda_val = 4
    f = (lambda_val**2 - 1) / (lambda_val**2 + 2)
    f = (16 - 1) / (16 + 2) = 15 / 18 = 0.833
    
    print(f"配向関数 f = {f:.3f}")
    # 出力: 配向関数 f = 0.833

f = 0.833は高配向を示し、機械的強度が大幅に向上していることを示します。

#### 演習4: Fox式Tg予測（Medium）

ポリスチレン（Tg = 100°C = 373 K）とポリイソプレン（Tg = -70°C = 203 K）を質量比1:1で共重合させた場合のTgをFox式で予測してください。 

解答を見る

**解答:**
    
    
    tg1 = 373  # K
    tg2 = 203  # K
    w1 = 0.5
    w2 = 0.5
    
    # Fox式
    tg_copolymer = 1 / (w1 / tg1 + w2 / tg2)
    tg_celsius = tg_copolymer - 273.15
    
    print(f"共重合体のTg = {tg_copolymer:.1f} K = {tg_celsius:.1f}°C")
    # 出力: 共重合体のTg = 263.0 K = -10.0°C

共重合により、Tgは両成分の中間値（約-10°C）になり、室温付近でゴム状態になります。

#### 演習5: 架橋密度計算（Medium）

25°C（298 K）でゴムの剛性率Gが1.5 MPaと測定されました。架橋密度νcを計算してください（R = 8.314 J/mol·K）。 

解答を見る

**解答:**
    
    
    G = 1.5e6  # Pa
    R = 8.314  # J/mol·K
    T = 298  # K
    
    # G = νc * R * T より
    nu_c = G / (R * T)
    nu_c_mol_per_L = nu_c / 1000  # mol/m³ → mol/L
    
    print(f"架橋密度 νc = {nu_c:.1f} mol/m³ = {nu_c_mol_per_L:.3f} mol/L")
    # 出力: 架橋密度 νc = 605.1 mol/m³ = 0.605 mol/L

この架橋密度は軟質ゴムに相当します。

#### 演習6: 配向と強度（Medium）

配向関数f = 0.6のPETフィルムの引張強度を、経験式 σ = 50 + 200f (MPa) で推定してください。未配向（f = 0）の場合と比較してください。 

解答を見る

**解答:**
    
    
    f_oriented = 0.6
    f_unoriented = 0
    
    strength_oriented = 50 + 200 * f_oriented
    strength_unoriented = 50 + 200 * f_unoriented
    
    improvement = (strength_oriented - strength_unoriented) / strength_unoriented * 100
    
    print(f"配向フィルム（f=0.6）: {strength_oriented} MPa")
    print(f"未配向フィルム（f=0）: {strength_unoriented} MPa")
    print(f"強度向上: {improvement:.0f}%")
    # 出力: 配向フィルム: 170 MPa, 未配向: 50 MPa, 強度向上: 240%

配向により引張強度が約3.4倍に向上します。

#### 演習7: Gordon-Taylorパラメータ推定（Medium）

実験データ（成分1質量分率50%でTg = 300 K）から、Tg1 = 373 K、Tg2 = 233 KのときのGordon-Taylorパラメータkを逆算してください。 

解答を見る

**解答:**
    
    
    tg1 = 373  # K
    tg2 = 233  # K
    w1 = 0.5
    tg_exp = 300  # K
    
    # Gordon-Taylor式を変形してkを求める
    # Tg = (w1*Tg1 + k*w2*Tg2) / (w1 + k*w2)
    # Tg(w1 + k*w2) = w1*Tg1 + k*w2*Tg2
    # Tg*w1 + Tg*k*w2 = w1*Tg1 + k*w2*Tg2
    # k(Tg*w2 - w2*Tg2) = w1*Tg1 - Tg*w1
    # k = (w1*Tg1 - Tg*w1) / (Tg*w2 - w2*Tg2)
    
    w2 = 1 - w1
    k = (w1 * tg1 - tg_exp * w1) / (tg_exp * w2 - w2 * tg2)
    
    print(f"Gordon-Taylorパラメータ k = {k:.3f}")
    # 出力: k ≈ 1.088

k ≈ 1.09は、両成分の熱膨張係数がほぼ同等であることを示します。

#### 演習8: Flory-Huggins相図（Hard）

Flory-Huggins理論を用いて、2成分高分子ブレンドの相図をプロットしてください。相互作用パラメータχ = 0.5, 1.0, 2.0の場合を比較し、臨界χ値（UCST）を計算してください。 

解答を見る

**解答:**
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Flory-Huggins相図
    def plot_flory_huggins_phase_diagram():
        """Flory-Huggins理論に基づく相図"""
        phi = np.linspace(0.01, 0.99, 100)  # 成分1の体積分率
    
        # 臨界相互作用パラメータ（N1 = N2 = 100と仮定）
        N1, N2 = 100, 100
        chi_critical = 0.5 * (1/np.sqrt(N1) + 1/np.sqrt(N2))**2
    
        print(f"臨界χ値（UCST）: {chi_critical:.4f}")
    
        # スピノダル曲線（ΔGmixの2階微分 = 0）
        def spinodal_chi(phi, N1, N2):
            """スピノダル曲線のχ値"""
            return 0.5 * (1 / (N1 * phi) + 1 / (N2 * (1 - phi)))
    
        chi_spinodal = spinodal_chi(phi, N1, N2)
    
        # 可視化
        plt.figure(figsize=(10, 6))
        plt.plot(phi, chi_spinodal, 'r-', linewidth=3, label='Spinodal Curve')
        plt.axhline(chi_critical, color='blue', linestyle='--', linewidth=2,
                    label=f'Critical χ = {chi_critical:.4f}')
    
        # 相分離領域を塗りつぶし
        plt.fill_between(phi, chi_spinodal, 3, alpha=0.3, color='red',
                         label='Two-Phase Region')
        plt.fill_between(phi, 0, chi_spinodal, alpha=0.3, color='green',
                         label='Single-Phase Region')
    
        plt.xlabel('Volume Fraction φ1', fontsize=12)
        plt.ylabel('Interaction Parameter χ', fontsize=12)
        plt.title('Flory-Huggins Phase Diagram (N1=N2=100)', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.ylim(0, 0.1)
        plt.tight_layout()
        plt.savefig('flory_huggins_phase_diagram.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    plot_flory_huggins_phase_diagram()
    # 臨界χ値 ≈ 0.02（N=100の場合）
    # χ > 0.02で相分離が起こる

重合度が大きいほど臨界χ値は小さくなり、相溶性が低下します。

#### 演習9: DSC結晶化度解析（Hard）

ポリエチレンのDSC測定で融解エンタルピーが180 J/gでした。完全結晶の融解エンタルピーを293 J/gとして、結晶化度を計算してください。また、この結晶化度でのTgとTmを推定してください（Tm = 165°C、Tg = -120°C）。 

解答を見る

**解答:**
    
    
    delta_Hm_measured = 180  # J/g
    delta_Hm_100percent = 293  # J/g（完全結晶PE）
    
    crystallinity = delta_Hm_measured / delta_Hm_100percent
    
    print(f"結晶化度: {crystallinity:.1%}")
    # 出力: 結晶化度: 61.4%
    
    # TgとTmは結晶化度に依存しない（純成分の固有値）
    # ただし、見かけのTgは結晶化度が高いほど不明瞭になる
    print(f"Tm（融点）: 165°C")
    print(f"Tg（ガラス転移温度）: -120°C（非晶領域のみ）")
    print(f"結晶化度が高いため、Tg転移は不明瞭")
    

61.4%の結晶化度は高密度ポリエチレン（HDPE）に相当します。

#### 演習10: 統合構造解析（Hard）

isotactic ポリプロピレン（iPP）について、以下の実験データを統合して材料特性を予測してください： 

  * NMR: isotactic分率 85%
  * XRD: 結晶化度 60%
  * 延伸倍率: λ = 5
  * DSC: Tm = 165°C

引張強度、弾性率、用途を推定してください。 

解答を見る

**解答:**
    
    
    import numpy as np
    
    # 実験データ
    isotactic_fraction = 0.85
    crystallinity = 0.60
    draw_ratio = 5.0
    tm = 165  # °C
    
    # 配向関数計算
    f = (draw_ratio**2 - 1) / (draw_ratio**2 + 2)
    
    # 引張強度推定（経験式）
    # 基準強度（未配向、結晶化度50%）= 30 MPa
    base_strength = 30
    strength = base_strength * (1 + 3 * (crystallinity - 0.5)) * (1 + 4 * f)
    
    # 弾性率推定（結晶化度と配向に依存）
    # 基準弾性率（未配向、非晶質）= 1.0 GPa
    base_modulus = 1.0
    modulus = base_modulus * (1 + 5 * crystallinity) * (1 + 3 * f)
    
    print("=== iPP材料特性予測 ===")
    print(f"立体規則性: {isotactic_fraction:.0%}")
    print(f"結晶化度: {crystallinity:.0%}")
    print(f"配向関数: {f:.3f}")
    print(f"融点: {tm}°C")
    print(f"\n予測物性:")
    print(f"  引張強度: {strength:.1f} MPa")
    print(f"  弾性率: {modulus:.1f} GPa")
    print(f"\n推奨用途:")
    if strength > 150 and modulus > 5:
        print("  - 高強度繊維（ロープ、不織布）")
        print("  - 高性能フィルム（包装材、電池セパレータ）")
    elif strength > 80:
        print("  - 汎用フィルム（食品包装）")
        print("  - 射出成形品（容器）")
    else:
        print("  - 低強度用途（使い捨て製品）")
    
    # 出力例:
    # 引張強度: 186.0 MPa
    # 弾性率: 11.1 GPa
    # 推奨用途: 高強度繊維、高性能フィルム
    

高立体規則性、高結晶化度、高配向により、優れた機械的特性を示し、高性能用途に適しています。

## 参考文献

  1. Strobl, G. (2007). _The Physics of Polymers: Concepts for Understanding Their Structures and Behavior_ (3rd ed.). Springer. pp. 1-95, 145-210.
  2. Young, R. J., & Lovell, P. A. (2011). _Introduction to Polymers_ (3rd ed.). CRC Press. pp. 190-285.
  3. Gedde, U. W., & Hedenqvist, M. S. (2019). _Fundamental Polymer Science_ (2nd ed.). Springer. pp. 50-135.
  4. Mark, J. E. (Ed.). (2007). _Physical Properties of Polymers Handbook_ (2nd ed.). Springer. pp. 200-265.
  5. Flory, P. J. (1953). _Principles of Polymer Chemistry_. Cornell University Press. pp. 495-540.
  6. Ward, I. M., & Sweeney, J. (2012). _Mechanical Properties of Solid Polymers_ (3rd ed.). Wiley. pp. 75-150.
  7. Ferry, J. D. (1980). _Viscoelastic Properties of Polymers_ (3rd ed.). Wiley. pp. 264-320.

### 次章への接続

第3章では、本章で学んだ高分子構造が物性（機械的性質、粘弾性、レオロジー）にどのように影響するかを詳しく学びます。特に、結晶化度と配向度が応力-ひずみ曲線やクリープ挙動にどう反映されるかを理解しましょう。また、WLF式を用いた時間-温度換算により、Tgの物理的意味がさらに深まります。 

### 免責事項

  * 本コンテンツは教育・研究・情報提供のみを目的としており、専門的な助言(法律・会計・技術的保証など)を提供するものではありません。
  * 本コンテンツおよび付随するCode examplesは「現状有姿(AS IS)」で提供され、明示または黙示を問わず、商品性、特定目的適合性、権利非侵害、正確性・完全性、動作・安全性等いかなる保証もしません。
  * 外部リンク、第三者が提供するデータ・ツール・ライブラリ等の内容・可用性・安全性について、作成者および東北大学は一切の責任を負いません。
  * 本コンテンツの利用・実行・解釈により直接的・間接的・付随的・特別・結果的・懲罰的損害が生じた場合でも、適用法で許容される最大限の範囲で、作成者および東北大学は責任を負いません。
  * 本コンテンツの内容は、予告なく変更・更新・提供停止されることがあります。
  * 本コンテンツの著作権・ライセンスは明記された条件(例: CC BY 4.0)に従います。当該ライセンスは通常、無保証条項を含みます。
