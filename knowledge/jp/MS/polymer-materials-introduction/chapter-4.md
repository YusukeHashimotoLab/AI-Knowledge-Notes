---
title: "第4章: 機能性高分子"
chapter_title: "第4章: 機能性高分子"
---

🌐 JP | [🇬🇧 EN](<../../../en/MS/polymer-materials-introduction/chapter-4.html>) | Last sync: 2025-11-16

[AI寺子屋トップ](<../../index.html>)›[材料科学](<../../MS/index.html>)›[Polymer Materials](<../../MS/polymer-materials-introduction/index.html>)›Chapter 4

  * [目次](<index.html>)
  * [← 第3章](<chapter-3.html>)
  * 第4章（準備中）
  * [第5章 →](<chapter-5.html>)

### 学習目標

**初級:**

  * 導電性高分子の基本メカニズム（π共役系とドーピング）を理解する
  * 生体適合性高分子の要件と代表例を説明できる
  * 刺激応答性高分子（温度、pH応答）の原理を理解する

**中級:**

  * 導電率とドーピングレベルの関係を計算できる
  * LCST（下限臨界共溶温度）をFlory-Huggins理論で予測できる
  * 薬物放出速度をカイネティクスモデルで解析できる

**上級:**

  * バンドギャップを計算し、光吸収スペクトルを予測できる
  * イオン伝導度をArrhenius式で解析できる
  * 生分解速度をモデル化し、分解時間を予測できる

## 4.1 導電性高分子

**導電性高分子（Conducting Polymers）** は、共役π電子系を持ち、ドーピングにより電気伝導性を示す有機材料です。代表例として、**ポリアニリン（PANI）** 、**PEDOT:PSS** 、**ポリピロール（PPy）** があります。 
    
    
    ```mermaid
    flowchart TD
                        A[導電性高分子] --> B[π共役系]
                        B --> C[HOMO-LUMOバンドギャップ]
                        A --> D[ドーピング]
                        D --> E[酸化ドープ p型 ]
                        D --> F[還元ドープ n型 ]
                        E --> G[ポーラロン形成]
                        G --> H[電荷キャリア生成]
                        H --> I[電気伝導性σ = 1-1000 S/cm]
                        I --> J[用途: 透明電極有機EL, 太陽電池]
    ```

### 4.1.1 導電率とドーピング

導電率σは、ドーピングレベル（酸化/還元度）に依存します。以下では、ポリアニリンのドーピングシミュレーションを行います。 
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    # 導電性高分子のドーピングシミュレーション
    def simulate_conductivity_doping(polymer='Polyaniline'):
        """
        ドーピングレベルと導電率の関係をシミュレート
    
        Parameters:
        - polymer: 高分子名（'Polyaniline', 'PEDOT', 'Polypyrrole'）
    
        Returns:
        - doping_levels: ドーピングレベル（%）
        - conductivities: 導電率（S/cm）
        """
        # ドーピングレベル範囲（0-50%）
        doping_levels = np.linspace(0, 50, 100)
    
        # 導電率モデル（経験式）
        # σ = σ_max * (x / x_opt)^2 * exp(-((x - x_opt) / w)^2)
        # x: ドーピングレベル, x_opt: 最適ドーピング, w: 幅パラメータ
    
        polymer_params = {
            'Polyaniline': {'sigma_max': 200, 'x_opt': 25, 'w': 15},
            'PEDOT': {'sigma_max': 1000, 'x_opt': 30, 'w': 20},
            'Polypyrrole': {'sigma_max': 100, 'x_opt': 20, 'w': 12}
        }
    
        params = polymer_params.get(polymer, polymer_params['Polyaniline'])
    
        # 導電率計算（S/cm）
        x_opt = params['x_opt']
        w = params['w']
        sigma_max = params['sigma_max']
    
        conductivities = sigma_max * ((doping_levels / x_opt) ** 2) * \
                         np.exp(-((doping_levels - x_opt) / w) ** 2)
    
        # 可視化
        plt.figure(figsize=(14, 5))
    
        # サブプロット1：導電率 vs ドーピングレベル
        plt.subplot(1, 3, 1)
        for poly_name, poly_params in polymer_params.items():
            x_opt_p = poly_params['x_opt']
            w_p = poly_params['w']
            sigma_max_p = poly_params['sigma_max']
            sigma_p = sigma_max_p * ((doping_levels / x_opt_p) ** 2) * \
                      np.exp(-((doping_levels - x_opt_p) / w_p) ** 2)
            plt.plot(doping_levels, sigma_p, linewidth=2, label=poly_name)
    
        plt.xlabel('Doping Level (%)', fontsize=12)
        plt.ylabel('Conductivity σ (S/cm)', fontsize=12)
        plt.title('Conductivity vs Doping Level', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.yscale('log')
    
        # サブプロット2：キャリア密度
        plt.subplot(1, 3, 2)
        # キャリア密度 n ∝ ドーピングレベル
        carrier_density = doping_levels * 1e20  # cm^-3（仮想値）
        mobility = conductivities / (1.6e-19 * carrier_density + 1e-10)  # cm²/V·s
    
        plt.plot(doping_levels, carrier_density / 1e20, 'b-', linewidth=2)
        plt.xlabel('Doping Level (%)', fontsize=12)
        plt.ylabel('Carrier Density n (×10²⁰ cm⁻³)', fontsize=12)
        plt.title(f'{polymer}: Carrier Density', fontsize=14, fontweight='bold')
        plt.grid(alpha=0.3)
    
        # サブプロット3：移動度
        plt.subplot(1, 3, 3)
        plt.plot(doping_levels, mobility, 'r-', linewidth=2)
        plt.xlabel('Doping Level (%)', fontsize=12)
        plt.ylabel('Mobility μ (cm²/V·s)', fontsize=12)
        plt.title(f'{polymer}: Charge Carrier Mobility', fontsize=14, fontweight='bold')
        plt.grid(alpha=0.3)
        plt.yscale('log')
    
        plt.tight_layout()
        plt.savefig('conductivity_doping.png', dpi=300, bbox_inches='tight')
        plt.show()
    
        # 結果出力
        print(f"=== {polymer}のドーピング解析 ===")
        print(f"最大導電率: {sigma_max} S/cm")
        print(f"最適ドーピングレベル: {x_opt}%")
    
        # 特定ドーピングレベルでの値
        for doping in [10, 25, 40]:
            idx = np.argmin(np.abs(doping_levels - doping))
            print(f"\nドーピングレベル {doping}%:")
            print(f"  導電率: {conductivities[idx]:.2f} S/cm")
            print(f"  キャリア密度: {carrier_density[idx]:.2e} cm⁻³")
    
        return doping_levels, conductivities
    
    # 実行
    simulate_conductivity_doping('Polyaniline')
    

### 4.1.2 バンドギャップ計算

共役高分子のバンドギャップEgは、光吸収スペクトルから決定できます。以下では、HOMOーLUMOギャップと光吸収の関係をシミュレートします。 
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    # バンドギャップと光吸収スペクトル
    def calculate_bandgap_absorption(bandgap_eV=2.5):
        """
        バンドギャップから光吸収スペクトルを計算
    
        Parameters:
        - bandgap_eV: バンドギャップエネルギー（eV）
    
        Returns:
        - wavelengths: 波長（nm）
        - absorbance: 吸光度
        """
        # 波長範囲（nm）
        wavelengths = np.linspace(300, 800, 500)
    
        # エネルギー変換 E(eV) = 1240 / λ(nm)
        photon_energies = 1240 / wavelengths
    
        # 吸収スペクトル（単純化：階段関数 + ガウシアンブロードニング）
        def absorption_profile(E, Eg, width=0.3):
            """吸収スペクトル（Gaussianブロードニング）"""
            if E < Eg:
                return 0
            else:
                return np.exp(-((E - Eg) / width) ** 2)
    
        absorbance = np.array([absorption_profile(E, bandgap_eV) for E in photon_energies])
    
        # 可視化
        plt.figure(figsize=(14, 5))
    
        # サブプロット1：吸収スペクトル
        plt.subplot(1, 3, 1)
        plt.plot(wavelengths, absorbance, 'b-', linewidth=2)
        # バンドギャップ対応波長
        lambda_g = 1240 / bandgap_eV
        plt.axvline(lambda_g, color='red', linestyle='--', linewidth=1.5,
                    label=f'λg = {lambda_g:.0f} nm (Eg = {bandgap_eV} eV)')
        plt.xlabel('Wavelength (nm)', fontsize=12)
        plt.ylabel('Absorbance (a.u.)', fontsize=12)
        plt.title('UV-Vis Absorption Spectrum', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(alpha=0.3)
    
        # 可視光領域を色分け
        plt.fill_betweenx([0, max(absorbance)], 380, 450, color='violet', alpha=0.2)
        plt.fill_betweenx([0, max(absorbance)], 450, 495, color='blue', alpha=0.2)
        plt.fill_betweenx([0, max(absorbance)], 495, 570, color='green', alpha=0.2)
        plt.fill_betweenx([0, max(absorbance)], 570, 590, color='yellow', alpha=0.2)
        plt.fill_betweenx([0, max(absorbance)], 590, 620, color='orange', alpha=0.2)
        plt.fill_betweenx([0, max(absorbance)], 620, 750, color='red', alpha=0.2)
    
        # サブプロット2：バンドギャップと色の関係
        plt.subplot(1, 3, 2)
        bandgaps = np.linspace(1.5, 3.5, 50)
        lambda_gaps = 1240 / bandgaps
        colors_perceived = []
        for lam in lambda_gaps:
            if lam < 400:
                colors_perceived.append('UV')
            elif lam < 450:
                colors_perceived.append('Violet')
            elif lam < 495:
                colors_perceived.append('Blue')
            elif lam < 570:
                colors_perceived.append('Green')
            elif lam < 590:
                colors_perceived.append('Yellow')
            elif lam < 620:
                colors_perceived.append('Orange')
            elif lam < 750:
                colors_perceived.append('Red')
            else:
                colors_perceived.append('IR')
    
        plt.scatter(bandgaps, lambda_gaps, c=lambda_gaps, cmap='rainbow', s=50, edgecolors='black')
        plt.xlabel('Bandgap Eg (eV)', fontsize=12)
        plt.ylabel('Absorption Edge λg (nm)', fontsize=12)
        plt.title('Bandgap vs Absorption Wavelength', fontsize=14, fontweight='bold')
        plt.colorbar(label='Wavelength (nm)')
        plt.grid(alpha=0.3)
        plt.axhline(lambda_g, color='red', linestyle='--', alpha=0.7)
    
        # サブプロット3：複数のバンドギャップ比較
        plt.subplot(1, 3, 3)
        bandgaps_examples = [1.8, 2.5, 3.2]
        for Eg in bandgaps_examples:
            photon_E = 1240 / wavelengths
            abs_spec = np.array([absorption_profile(E, Eg, 0.3) for E in photon_E])
            plt.plot(wavelengths, abs_spec, linewidth=2, label=f'Eg = {Eg} eV')
    
        plt.xlabel('Wavelength (nm)', fontsize=12)
        plt.ylabel('Absorbance (a.u.)', fontsize=12)
        plt.title('Effect of Bandgap on Absorption', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.xlim(300, 800)
    
        plt.tight_layout()
        plt.savefig('bandgap_absorption.png', dpi=300, bbox_inches='tight')
        plt.show()
    
        # 結果出力
        print("=== バンドギャップ解析 ===")
        print(f"バンドギャップ: {bandgap_eV} eV")
        print(f"吸収端波長: {lambda_g:.1f} nm")
    
        if lambda_g < 400:
            color_range = "紫外（UV）"
        elif lambda_g < 750:
            color_range = "可視光"
        else:
            color_range = "赤外（IR）"
    
        print(f"吸収領域: {color_range}")
        print(f"\n導電性高分子の典型的Eg: 1.5-3.0 eV")
    
        return wavelengths, absorbance
    
    # 実行例：Eg = 2.5 eV（PEDOT:PSS相当）
    calculate_bandgap_absorption(bandgap_eV=2.5)
    

## 4.2 生体適合性高分子

**生体適合性高分子（Biocompatible Polymers）** は、生体組織と接触しても毒性や免疫反応を引き起こさない材料です。代表例は**PEG（ポリエチレングリコール）** 、**ポリ乳酸（PLA）** 、**PLGA（ポリ乳酸-グリコール酸共重合体）** です。 

### 4.2.1 薬物放出カイネティクス

生分解性高分子からの薬物放出は、拡散と分解の競合で決定されます。以下では、Korsmeyer-Peppasモデルで放出挙動を解析します。 
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    # 薬物放出カイネティクス
    def simulate_drug_release_kinetics(model='Korsmeyer-Peppas'):
        """
        生分解性高分子からの薬物放出をシミュレート
    
        Parameters:
        - model: 放出モデル（'Korsmeyer-Peppas', 'Higuchi', 'First-order'）
    
        Returns:
        - time: 時間（時間）
        - release_fraction: 累積放出率（%）
        """
        # 時間範囲（時間）
        time = np.linspace(0, 48, 500)
    
        # Korsmeyer-Peppasモデル: Mt/M∞ = k * t^n
        # n: 拡散指数（n=0.5: Fickian拡散, n=1.0: Case II, 0.5

### 4.2.2 生分解速度解析

ポリ乳酸（PLA）などの生分解性高分子は、加水分解により分解します。分子量低下は一次反応でモデル化できます。 
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    # 生分解速度シミュレーション
    def simulate_biodegradation(polymer='PLA', temperature=310):
        """
        生分解性高分子の分解速度をシミュレート
    
        Parameters:
        - polymer: 高分子名（'PLA', 'PLGA', 'PCL'）
        - temperature: 温度（K）
    
        Returns:
        - time: 時間（日）
        - molecular_weight: 分子量（相対値）
        """
        # 分解パラメータ（Arrhenius式）
        # k = k0 * exp(-Ea/RT)
        polymer_params = {
            'PLA': {'k0': 1e10, 'Ea': 80000},  # J/mol
            'PLGA': {'k0': 1e11, 'Ea': 75000},
            'PCL': {'k0': 1e9, 'Ea': 85000}
        }
    
        params = polymer_params.get(polymer, polymer_params['PLA'])
        k0 = params['k0']
        Ea = params['Ea']
        R = 8.314  # J/mol·K
    
        # 速度定数（1/日）
        k = k0 * np.exp(-Ea / (R * temperature)) * 86400  # 秒→日変換
    
        # 時間範囲（日）
        time = np.linspace(0, 365, 500)
    
        # 一次分解: Mw(t) = Mw0 * exp(-k*t)
        Mw0 = 100000  # 初期分子量（g/mol）
        molecular_weight = Mw0 * np.exp(-k * time)
    
        # 可視化
        plt.figure(figsize=(14, 5))
    
        # サブプロット1：分子量低下
        plt.subplot(1, 3, 1)
        for poly_name, poly_params in polymer_params.items():
            k_poly = poly_params['k0'] * np.exp(-poly_params['Ea'] / (R * temperature)) * 86400
            Mw_t = Mw0 * np.exp(-k_poly * time)
            plt.plot(time, Mw_t / 1000, linewidth=2, label=poly_name)
    
        plt.xlabel('Time (days)', fontsize=12)
        plt.ylabel('Molecular Weight (kDa)', fontsize=12)
        plt.title(f'Biodegradation at {temperature}K ({temperature-273.15:.0f}°C)', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.yscale('log')
    
        # サブプロット2：温度依存性
        plt.subplot(1, 3, 2)
        temperatures = [298, 310, 323]  # K（25, 37, 50°C）
        for T in temperatures:
            k_T = k0 * np.exp(-Ea / (R * T)) * 86400
            Mw_T = Mw0 * np.exp(-k_T * time)
            plt.plot(time, Mw_T / 1000, linewidth=2, label=f'{T-273.15:.0f}°C')
    
        plt.xlabel('Time (days)', fontsize=12)
        plt.ylabel('Molecular Weight (kDa)', fontsize=12)
        plt.title(f'{polymer}: Temperature Dependence', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.yscale('log')
    
        # サブプロット3：分解率（質量残存率）
        plt.subplot(1, 3, 3)
        # 分解率 = (Mw0 - Mw(t)) / Mw0 * 100
        degradation_percent = (1 - molecular_weight / Mw0) * 100
        plt.plot(time, degradation_percent, 'b-', linewidth=2)
        plt.axhline(50, color='red', linestyle='--', linewidth=1.5, label='50% Degradation')
        # 50%分解時間
        t_50 = -np.log(0.5) / k
        plt.axvline(t_50, color='green', linestyle='--', linewidth=1.5,
                    label=f't₅₀ = {t_50:.0f} days')
        plt.xlabel('Time (days)', fontsize=12)
        plt.ylabel('Degradation (%)', fontsize=12)
        plt.title(f'{polymer}: Degradation Progress', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(alpha=0.3)
    
        plt.tight_layout()
        plt.savefig('biodegradation.png', dpi=300, bbox_inches='tight')
        plt.show()
    
        # 結果出力
        print(f"=== {polymer}生分解解析（{temperature}K = {temperature-273.15:.0f}°C）===")
        print(f"初期分子量 Mw0: {Mw0/1000:.0f} kDa")
        print(f"活性化エネルギー Ea: {Ea/1000:.0f} kJ/mol")
        print(f"分解速度定数 k: {k:.2e} 1/day")
        print(f"50%分解時間 t₅₀: {t_50:.0f} 日")
        print(f"90%分解時間 t₉₀: {-np.log(0.1)/k:.0f} 日")
    
        return time, molecular_weight
    
    # 実行例：PLA、37°C（体温）
    simulate_biodegradation('PLA', temperature=310)
    

## 4.3 刺激応答性高分子

**刺激応答性高分子（Stimuli-Responsive Polymers）** は、温度、pH、光などの外部刺激に応答して構造や物性を変化させます。代表例は**PNIPAM（ポリN-イソプロピルアクリルアミド）** で、**LCST（下限臨界共溶温度）** を示します。 

### 4.3.1 LCST計算（Flory-Huggins理論）

LCSTは、Flory-Huggins相互作用パラメータχが温度依存性を持つことで説明されます： 

\\[ \chi = A + \frac{B}{T} \\] 
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    # LCST計算（Flory-Huggins理論）
    def calculate_lcst_flory_huggins(polymer='PNIPAM'):
        """
        Flory-Huggins理論に基づくLCST相図を計算
    
        Parameters:
        - polymer: 高分子名（'PNIPAM', 'PEO'）
    
        Returns:
        - temperatures: 温度（K）
        - volume_fractions: 体積分率
        """
        # Flory-Hugginsパラメータ（温度依存性）
        # χ(T) = A + B/T
        polymer_params = {
            'PNIPAM': {'A': -12.0, 'B': 4300},  # K
            'PEO': {'A': -15.0, 'B': 5000}
        }
    
        params = polymer_params.get(polymer, polymer_params['PNIPAM'])
        A = params['A']
        B = params['B']
    
        # 体積分率範囲
        phi = np.linspace(0.01, 0.99, 100)
    
        # 重合度（高分子/溶媒）
        N = 1000  # 高分子重合度
    
        # スピノダル曲線（二次微分 = 0）
        # d²ΔGmix/dφ² = 0 → χ_spinodal = 0.5 * (1/(N*φ) + 1/(1-φ))
        chi_spinodal = 0.5 * (1 / (N * phi) + 1 / (1 - phi))
    
        # 温度計算（χ = A + B/T から T = B / (χ - A)）
        temperatures_spinodal = B / (chi_spinodal - A)
    
        # 臨界点（φ = 1/√(N+1) ≈ 1/√N）
        phi_critical = 1 / np.sqrt(N + 1)
        chi_critical = 0.5 * (1 + 1/np.sqrt(N))**2
        T_critical = B / (chi_critical - A)
    
        # 可視化
        plt.figure(figsize=(14, 5))
    
        # サブプロット1：相図
        plt.subplot(1, 3, 1)
        plt.plot(phi * 100, temperatures_spinodal - 273.15, 'b-', linewidth=2, label='Spinodal Curve (LCST)')
        plt.scatter([phi_critical * 100], [T_critical - 273.15], s=200, c='red',
                    edgecolors='black', linewidths=2, zorder=5, label=f'Critical Point ({T_critical-273.15:.1f}°C)')
        plt.fill_between(phi * 100, temperatures_spinodal - 273.15, 100, alpha=0.3, color='red',
                         label='Two-Phase Region')
        plt.fill_between(phi * 100, 0, temperatures_spinodal - 273.15, alpha=0.3, color='green',
                         label='Single-Phase Region')
        plt.xlabel('Polymer Volume Fraction φ (%)', fontsize=12)
        plt.ylabel('Temperature (°C)', fontsize=12)
        plt.title(f'{polymer} LCST Phase Diagram', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.ylim(0, 100)
    
        # サブプロット2：χパラメータの温度依存性
        plt.subplot(1, 3, 2)
        T_range = np.linspace(273, 373, 100)  # K
        chi_T = A + B / T_range
        plt.plot(T_range - 273.15, chi_T, 'purple', linewidth=2)
        plt.axhline(chi_critical, color='red', linestyle='--', linewidth=1.5,
                    label=f'χ_crit = {chi_critical:.3f}')
        plt.axvline(T_critical - 273.15, color='green', linestyle='--', linewidth=1.5,
                    label=f'LCST = {T_critical-273.15:.1f}°C')
        plt.xlabel('Temperature (°C)', fontsize=12)
        plt.ylabel('Flory-Huggins Parameter χ', fontsize=12)
        plt.title('Temperature Dependence of χ', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(alpha=0.3)
    
        # サブプロット3：濁度変化シミュレーション
        plt.subplot(1, 3, 3)
        phi_sample = 0.05  # 5% 溶液
        temperatures_exp = np.linspace(10, 60, 100)  # °C
        chi_exp = A + B / (temperatures_exp + 273.15)
        # 相分離判定（χ > χ_spinodal で相分離）
        chi_spinodal_at_phi = 0.5 * (1 / (N * phi_sample) + 1 / (1 - phi_sample))
        turbidity = np.where(chi_exp > chi_spinodal_at_phi, 1, 0)
    
        plt.plot(temperatures_exp, turbidity, 'b-', linewidth=3)
        plt.fill_between(temperatures_exp, turbidity, alpha=0.3, color='blue')
        plt.axvline(T_critical - 273.15, color='red', linestyle='--', linewidth=1.5,
                    label=f'LCST = {T_critical-273.15:.1f}°C')
        plt.xlabel('Temperature (°C)', fontsize=12)
        plt.ylabel('Turbidity (Phase Separation)', fontsize=12)
        plt.title(f'{polymer} (φ = {phi_sample*100}%): Turbidity Change', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.ylim(-0.1, 1.2)
    
        plt.tight_layout()
        plt.savefig('lcst_phase_diagram.png', dpi=300, bbox_inches='tight')
        plt.show()
    
        # 結果出力
        print(f"=== {polymer} LCST解析（Flory-Huggins理論）===")
        print(f"Flory-Hugginsパラメータ: χ = {A} + {B}/T")
        print(f"重合度 N: {N}")
        print(f"臨界体積分率 φ_c: {phi_critical:.4f}")
        print(f"臨界χ値: {chi_critical:.4f}")
        print(f"LCST: {T_critical - 273.15:.1f}°C")
    
        return temperatures_spinodal, phi
    
    # 実行
    calculate_lcst_flory_huggins('PNIPAM')
    

### 4.3.2 pH応答性電離度計算
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    # pH応答性高分子の電離度計算
    def calculate_ph_responsive_ionization(pKa=5.5):
        """
        Henderson-Hasselbalch式によるpH応答性高分子の電離度を計算
    
        Parameters:
        - pKa: 酸解離定数
    
        Returns:
        - pH_values: pH値
        - ionization_degrees: 電離度
        """
        # pH範囲
        pH_values = np.linspace(2, 10, 200)
    
        # Henderson-Hasselbalch式
        # α = 1 / (1 + 10^(pKa - pH))（弱酸性基の場合）
        ionization_degree = 1 / (1 + 10**(pKa - pH_values))
    
        # 可視化
        plt.figure(figsize=(14, 5))
    
        # サブプロット1：電離度 vs pH
        plt.subplot(1, 3, 1)
        pKa_values = [4.5, 5.5, 6.5]
        for pKa_val in pKa_values:
            alpha = 1 / (1 + 10**(pKa_val - pH_values))
            plt.plot(pH_values, alpha * 100, linewidth=2, label=f'pKa = {pKa_val}')
    
        plt.axhline(50, color='gray', linestyle='--', linewidth=1, alpha=0.7)
        plt.xlabel('pH', fontsize=12)
        plt.ylabel('Ionization Degree (%)', fontsize=12)
        plt.title('pH-Responsive Ionization', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(alpha=0.3)
    
        # サブプロット2：膨潤比（電離度に比例）
        plt.subplot(1, 3, 2)
        # 膨潤比 Q ∝ α²（Donnan効果）
        swelling_ratio = 1 + 10 * ionization_degree**2
        plt.plot(pH_values, swelling_ratio, 'purple', linewidth=2)
        plt.axvline(pKa, color='red', linestyle='--', linewidth=1.5, label=f'pKa = {pKa}')
        plt.xlabel('pH', fontsize=12)
        plt.ylabel('Swelling Ratio Q/Q₀', fontsize=12)
        plt.title('pH-Induced Swelling', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(alpha=0.3)
    
        # サブプロット3：滴定曲線
        plt.subplot(1, 3, 3)
        # 滴定曲線（NaOH添加量と pH）
        # 簡略化：弱酸の滴定
        V_NaOH = np.linspace(0, 50, 200)  # mL
        # pH = pKa + log((V_NaOH) / (V_eq - V_NaOH))（V_eq: 当量点）
        V_eq = 25  # mL
        pH_titration = []
        for V in V_NaOH:
            if V < V_eq:
                if V > 0:
                    pH_val = pKa + np.log10(V / (V_eq - V))
                else:
                    pH_val = 3  # 初期pH（仮定）
            elif V == V_eq:
                pH_val = 7  # 当量点（弱酸-強塩基）
            else:
                pH_val = 7 + np.log10((V - V_eq) / V_eq)
            pH_titration.append(pH_val)
    
        plt.plot(V_NaOH, pH_titration, 'g-', linewidth=2)
        plt.axvline(V_eq, color='red', linestyle='--', linewidth=1.5, label=f'Equivalence Point ({V_eq} mL)')
        plt.axhline(pKa, color='blue', linestyle='--', linewidth=1.5, label=f'pKa = {pKa}')
        plt.xlabel('Volume of NaOH (mL)', fontsize=12)
        plt.ylabel('pH', fontsize=12)
        plt.title('Titration Curve of pH-Responsive Polymer', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.ylim(2, 12)
    
        plt.tight_layout()
        plt.savefig('ph_responsive_ionization.png', dpi=300, bbox_inches='tight')
        plt.show()
    
        # 結果出力
        print("=== pH応答性高分子解析 ===")
        print(f"pKa: {pKa}")
        print(f"pH = pKa での電離度: 50%")
        print(f"\npH値別電離度:")
        for pH_target in [3, 5, 7, 9]:
            idx = np.argmin(np.abs(pH_values - pH_target))
            print(f"  pH {pH_target}: {ionization_degree[idx]*100:.1f}%")
    
        return pH_values, ionization_degree
    
    # 実行
    calculate_ph_responsive_ionization(pKa=5.5)
    

## 4.4 高分子電解質

**高分子電解質（Polymer Electrolytes）** は、イオン伝導性を示す高分子材料で、リチウムイオン電池や燃料電池に応用されます。代表例は**Nafion** （プロトン伝導膜）です。 

### 4.4.1 イオン伝導度のArrhenius解析
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    # イオン伝導度のArrhenius解析
    def analyze_ionic_conductivity(polymer='Nafion'):
        """
        高分子電解質のイオン伝導度をArrhenius式で解析
    
        Parameters:
        - polymer: 高分子名（'Nafion', 'PEO-LiTFSI'）
    
        Returns:
        - temperatures: 温度（K）
        - conductivities: イオン伝導度（S/cm）
        """
        # Arrhenius式: σ = σ0 * exp(-Ea / RT)
        polymer_params = {
            'Nafion': {'sigma0': 1e4, 'Ea': 15000},  # S/cm, J/mol
            'PEO-LiTFSI': {'sigma0': 1e6, 'Ea': 50000}
        }
    
        params = polymer_params.get(polymer, polymer_params['Nafion'])
        sigma0 = params['sigma0']
        Ea = params['Ea']
        R = 8.314  # J/mol·K
    
        # 温度範囲（K）
        temperatures = np.linspace(273, 373, 100)
    
        # イオン伝導度（S/cm）
        conductivities = sigma0 * np.exp(-Ea / (R * temperatures))
    
        # 可視化
        plt.figure(figsize=(14, 5))
    
        # サブプロット1：Arrhenius プロット
        plt.subplot(1, 3, 1)
        for poly_name, poly_params in polymer_params.items():
            sigma0_p = poly_params['sigma0']
            Ea_p = poly_params['Ea']
            sigma_p = sigma0_p * np.exp(-Ea_p / (R * temperatures))
            plt.plot(1000 / temperatures, np.log10(sigma_p), linewidth=2, label=poly_name)
    
        plt.xlabel('1000/T (K⁻¹)', fontsize=12)
        plt.ylabel('log(σ) [S/cm]', fontsize=12)
        plt.title('Arrhenius Plot of Ionic Conductivity', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(alpha=0.3)
    
        # サブプロット2：導電率 vs 温度
        plt.subplot(1, 3, 2)
        plt.plot(temperatures - 273.15, conductivities, 'b-', linewidth=2, label=polymer)
        plt.axhline(1e-4, color='red', linestyle='--', linewidth=1.5,
                    label='Target (10⁻⁴ S/cm)')
        plt.xlabel('Temperature (°C)', fontsize=12)
        plt.ylabel('Ionic Conductivity σ (S/cm)', fontsize=12)
        plt.title(f'{polymer}: Conductivity vs Temperature', fontsize=14, fontweight='bold')
        plt.yscale('log')
        plt.legend()
        plt.grid(alpha=0.3)
    
        # サブプロット3：活性化エネルギー比較
        plt.subplot(1, 3, 3)
        poly_names = list(polymer_params.keys())
        Ea_values = [polymer_params[p]['Ea'] / 1000 for p in poly_names]  # kJ/mol
    
        bars = plt.bar(poly_names, Ea_values, color=['#4A90E2', '#E74C3C'],
                       edgecolor='black', linewidth=2)
        plt.ylabel('Activation Energy Ea (kJ/mol)', fontsize=12)
        plt.title('Comparison of Activation Energies', fontsize=14, fontweight='bold')
        plt.grid(alpha=0.3, axis='y')
    
        for bar, val in zip(bars, Ea_values):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.0f} kJ/mol', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
        plt.tight_layout()
        plt.savefig('ionic_conductivity.png', dpi=300, bbox_inches='tight')
        plt.show()
    
        # 結果出力
        print(f"=== {polymer}イオン伝導度解析 ===")
        print(f"Arrheniusパラメータ: σ0 = {sigma0:.2e} S/cm, Ea = {Ea/1000:.1f} kJ/mol")
    
        for T_target in [298, 323, 353]:  # 25, 50, 80°C
            idx = np.argmin(np.abs(temperatures - T_target))
            print(f"\n温度 {T_target}K ({T_target-273.15:.0f}°C):")
            print(f"  イオン伝導度: {conductivities[idx]:.2e} S/cm")
    
        return temperatures, conductivities
    
    # 実行
    analyze_ionic_conductivity('Nafion')
    

## 演習問題

#### 演習1: 導電率計算（Easy）

ドーピングレベル30%、最大導電率500 S/cm、最適ドーピング25%のとき、導電率を簡易式σ = σ_max × (x/x_opt)で計算してください。

解答を見る
    
    
    sigma_max = 500
    x = 30
    x_opt = 25
    sigma = sigma_max * (x / x_opt)
    print(f"導電率: {sigma} S/cm")
    # 出力: 600 S/cm（過剰ドーピング）

#### 演習2: バンドギャップ計算（Easy）

吸収端波長が550 nmの導電性高分子のバンドギャップ（eV）を計算してください。

解答を見る
    
    
    lambda_nm = 550
    Eg = 1240 / lambda_nm
    print(f"バンドギャップ: {Eg:.2f} eV")
    # 出力: 2.25 eV

#### 演習3: 薬物放出時間（Easy）

Korsmeyer-Peppasモデル（k=0.1, n=0.5）で50%放出時間を計算してください。

解答を見る
    
    
    k = 0.1
    n = 0.5
    Mt_Minf = 0.5
    t_50 = (Mt_Minf / k)**(1/n)
    print(f"50%放出時間: {t_50:.1f} 時間")
    # 出力: 25.0 時間

#### 演習4: LCST予測（Medium）

χ = -12 + 4300/T、臨界χ = 0.502のとき、LCSTを計算してください。

解答を見る
    
    
    A = -12
    B = 4300
    chi_critical = 0.502
    T_lcst = B / (chi_critical - A)
    print(f"LCST: {T_lcst:.1f} K = {T_lcst - 273.15:.1f}°C")
    # 出力: LCST: 344.0 K = 70.8°C

#### 演習5: pH応答電離度（Medium）

pKa = 5.5の高分子をpH 7.0の溶液に浸漬したとき、電離度を計算してください。

解答を見る
    
    
    pKa = 5.5
    pH = 7.0
    alpha = 1 / (1 + 10**(pKa - pH))
    print(f"電離度: {alpha*100:.1f}%")
    # 出力: 96.9%

#### 演習6: 生分解半減期（Medium）

速度定数k = 0.005 1/dayのとき、分子量半減期を計算してください。

解答を見る
    
    
    import numpy as np
    k = 0.005
    t_half = np.log(2) / k
    print(f"半減期: {t_half:.0f} 日")
    # 出力: 139 日

#### 演習7: イオン伝導度計算（Medium）

σ0 = 1×10⁴ S/cm、Ea = 15 kJ/mol、T = 80°C（353 K）のとき、イオン伝導度を計算してください（R = 8.314 J/mol·K）。

解答を見る
    
    
    import numpy as np
    sigma0 = 1e4
    Ea = 15000
    R = 8.314
    T = 353
    sigma = sigma0 * np.exp(-Ea / (R * T))
    print(f"イオン伝導度: {sigma:.2e} S/cm")
    # 出力: 約 0.01 S/cm

#### 演習8: 光吸収スペクトル予測（Hard）

Eg = 2.0 eVの共役高分子について、吸収端波長と主な吸収色を予測してください。また、透過光の色を推定してください。

解答を見る
    
    
    Eg = 2.0
    lambda_edge = 1240 / Eg
    print(f"吸収端波長: {lambda_edge:.0f} nm")
    print("吸収色: 赤〜緑（波長620nm以下）")
    print("透過光の色: 赤（補色として赤が透過）")
    # 出力: 620 nm、赤色領域まで吸収、赤色を呈する

#### 演習9: 薬物放出制御（Hard）

24時間で80%放出を達成したい。Korsmeyer-Peppasモデル（n=0.6）でパラメータkを最適化してください。

解答を見る
    
    
    import numpy as np
    t_target = 24
    Mt_Minf_target = 0.8
    n = 0.6
    k = Mt_Minf_target / (t_target**n)
    print(f"最適k: {k:.4f}")
    print(f"検証: Mt/M∞ = {k * (24**n):.2f}")
    # 出力: k ≈ 0.0927, 24時間後80%放出

#### 演習10: 多機能性高分子設計（Hard）

導電性（σ > 1 S/cm）と生体適合性を両立する高分子を設計してください。PEDOT-PEGコポリマーを想定し、最適組成を提案してください。

解答を見る

**設計方針:**

  * PEDOT含量: 70-80%（導電性確保）
  * PEG含量: 20-30%（生体適合性・親水性付与）
  * 期待特性: σ = 1-10 S/cm、細胞接着性良好

    
    
    # 最適組成シミュレーション
    PEDOT_ratio = 0.75
    PEG_ratio = 0.25
    sigma_max_PEDOT = 1000  # S/cm
    sigma_estimated = sigma_max_PEDOT * PEDOT_ratio * 0.1  # 希釈効果考慮
    print(f"PEDOT: {PEDOT_ratio*100}%, PEG: {PEG_ratio*100}%")
    print(f"予測導電率: {sigma_estimated:.1f} S/cm")
    print("生体適合性: PEGにより細胞接着性向上")
    # 出力: σ ≈ 7.5 S/cm、生体適合性良好

## 参考文献

  1. Skotheim, T. A., & Reynolds, J. R. (Eds.). (2007). _Handbook of Conducting Polymers_ (3rd ed.). CRC Press. pp. 1-85.
  2. Ratner, B. D., et al. (2013). _Biomaterials Science: An Introduction to Materials in Medicine_ (3rd ed.). Academic Press. pp. 120-195.
  3. Stuart, M. A. C., et al. (2010). Emerging applications of stimuli-responsive polymer materials. _Nature Materials_ , 9, 101-113.
  4. Dobrynin, A. V., & Rubinstein, M. (2005). Theory of polyelectrolytes in solutions and at surfaces. _Progress in Polymer Science_ , 30, 1049-1118.
  5. Mauritz, K. A., & Moore, R. B. (2004). State of understanding of Nafion. _Chemical Reviews_ , 104(10), 4535-4585.
  6. Siepmann, J., & Peppas, N. A. (2001). Modeling of drug release from delivery systems. _Advanced Drug Delivery Reviews_ , 48, 139-157.

### 次章への接続

第5章では、本シリーズで学んだ全ての知識を統合し、Pythonによる実践的ワークフローを構築します。RDKitによる高分子構造生成、機械学習によるTg予測、MDシミュレーションデータ解析、そしてPolyInfoなどのデータベース連携まで、実務で即戦力となるスキルを習得します。 

### 免責事項

  * 本コンテンツは教育・研究・情報提供のみを目的としており、専門的な助言(法律・会計・技術的保証など)を提供するものではありません。
  * 本コンテンツおよび付随するCode examplesは「現状有姿(AS IS)」で提供され、明示または黙示を問わず、商品性、特定目的適合性、権利非侵害、正確性・完全性、動作・安全性等いかなる保証もしません。
  * 外部リンク、第三者が提供するデータ・ツール・ライブラリ等の内容・可用性・安全性について、作成者および東北大学は一切の責任を負いません。
  * 本コンテンツの利用・実行・解釈により直接的・間接的・付随的・特別・結果的・懲罰的損害が生じた場合でも、適用法で許容される最大限の範囲で、作成者および東北大学は責任を負いません。
  * 本コンテンツの内容は、予告なく変更・更新・提供停止されることがあります。
  * 本コンテンツの著作権・ライセンスは明記された条件(例: CC BY 4.0)に従います。当該ライセンスは通常、無保証条項を含みます。
