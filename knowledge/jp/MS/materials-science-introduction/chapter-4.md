---
title: 第4章：材料の性質と構造の関係
chapter_title: 第4章：材料の性質と構造の関係
subtitle: 機械的・電気的・熱的・光学的性質の理解
reading_time: 35-40分
difficulty: 中級
code_examples: 6
---

材料の性質は、原子構造・結晶構造・化学結合の種類によって決まります。この章では、機械的性質（強度・硬度・延性）、電気的性質（導電性・半導性）、熱的性質（熱伝導・熱膨張）、光学的性質（透明性・色）について学び、Pythonで材料特性を計算・可視化します。 

## 学習目標

この章を読むことで、以下を習得できます：

  * ✅ 応力-ひずみ曲線を理解し、材料の機械的性質を評価できる
  * ✅ 硬度測定法（Vickers, Brinell, Rockwell）を理解し、換算できる
  * ✅ バンド構造と電気伝導の関係を説明できる
  * ✅ 熱的性質（熱伝導率、線膨張係数、比熱）を理解する
  * ✅ 光学的性質（透明性、色、屈折率）の基礎を理解する
  * ✅ Pythonで材料特性を計算・プロット・比較できる

* * *

## 4.1 機械的性質（強度、硬度、延性）

### 応力とひずみ

**応力（Stress）** は、材料の単位面積あたりに作用する力です：

$$\sigma = \frac{F}{A}$$

ここで、

  * $\sigma$: 応力（Pa = N/m² または MPa）
  * $F$: 荷重（N）
  * $A$: 断面積（m²）

**ひずみ（Strain）** は、材料の相対的な変形量です：

$$\varepsilon = \frac{\Delta L}{L_0}$$

ここで、

  * $\varepsilon$: ひずみ（無次元）
  * $\Delta L$: 伸び量（m）
  * $L_0$: 元の長さ（m）

### 応力-ひずみ曲線

材料に荷重をかけて引っ張ると、応力-ひずみ曲線が得られます。この曲線から材料の機械的性質がわかります。

**主要な領域** ：

  1. **弾性領域（Elastic Region）** ：フックの法則に従う線形領域。荷重を除くと元に戻る。
  2. **降伏点（Yield Point）** ：弾性から塑性へ移行する点。降伏強度（Yield Strength）で評価。
  3. **塑性領域（Plastic Region）** ：永久変形が生じる領域。
  4. **引張強度（Ultimate Tensile Strength, UTS）** ：材料が耐えられる最大応力。
  5. **破断（Fracture）** ：材料が破壊される点。

**ヤング率（Young's Modulus, E）** ：弾性領域の傾き

$$E = \frac{\sigma}{\varepsilon}$$

単位はGPa（ギガパスカル）です。ヤング率が大きいほど、材料は硬く、変形しにくいです。

材料 | ヤング率（GPa） | 降伏強度（MPa） | 引張強度（MPa） | 延性  
---|---|---|---|---  
**鋼（Steel）** | 200 | 250-400 | 400-550 | 高  
**アルミニウム（Al）** | 69 | 35-100 | 90-150 | 高  
**銅（Cu）** | 130 | 70 | 220 | 高  
**チタン（Ti）** | 116 | 140-500 | 240-550 | 中  
**ガラス（SiO₂）** | 70 | - | 50-100 | 脆性  
**セラミックス（Al₂O₃）** | 380 | - | 300-400 | 脆性  
  
### 延性と脆性

**延性材料（Ductile）** ：塑性変形が大きい（金属）

  * 破断前に大きく伸びる
  * 破断伸び率が高い（通常 > 5%）
  * 例：銅、アルミニウム、鋼

**脆性材料（Brittle）** ：塑性変形が小さい（セラミックス、ガラス）

  * ほとんど伸びずに破断
  * 破断伸び率が低い（通常 < 5%）
  * 例：ガラス、セラミックス、鋳鉄

### コード例1: 応力-ひずみ曲線の作成とプロット（複数材料比較）

鋼、アルミニウム、ガラスの応力-ひずみ曲線を作成し、材料の違いを可視化します。
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    # 応力-ひずみ曲線のシミュレーション
    def stress_strain_curve(material='steel'):
        """
        材料の応力-ひずみ曲線をシミュレート
    
        Parameters:
        material: 'steel', 'aluminum', 'glass'のいずれか
    
        Returns:
        strain: ひずみ配列
        stress: 応力配列（MPa）
        """
        if material == 'steel':
            # 鋼（延性材料）
            E = 200e3  # ヤング率（MPa）
            yield_stress = 250  # 降伏強度（MPa）
            yield_strain = yield_stress / E  # 降伏ひずみ
            uts = 400  # 引張強度（MPa）
            fracture_strain = 0.25  # 破断ひずみ（25%伸び）
    
            # 弾性領域（0から降伏点まで）
            strain_elastic = np.linspace(0, yield_strain, 100)
            stress_elastic = E * strain_elastic
    
            # 塑性領域（降伏点から破断まで）- 加工硬化を考慮
            strain_plastic = np.linspace(yield_strain, fracture_strain, 300)
            # 加工硬化: 応力が増加するが、増加率は減少
            stress_plastic = yield_stress + (uts - yield_stress) * \
                            (1 - np.exp(-10 * (strain_plastic - yield_strain)))
    
            # ネッキング後の軟化
            strain_necking = np.linspace(fracture_strain, fracture_strain + 0.05, 50)
            stress_necking = uts * np.exp(-20 * (strain_necking - fracture_strain))
    
            strain = np.concatenate([strain_elastic, strain_plastic, strain_necking])
            stress = np.concatenate([stress_elastic, stress_plastic, stress_necking])
    
            properties = {
                'E': E,
                'yield_stress': yield_stress,
                'UTS': uts,
                'fracture_strain': fracture_strain + 0.05,
                'type': '延性材料'
            }
    
        elif material == 'aluminum':
            # アルミニウム（延性材料、鋼より柔らかい）
            E = 69e3  # ヤング率（MPa）
            yield_stress = 35  # 降伏強度（MPa）
            yield_strain = yield_stress / E
            uts = 90  # 引張強度（MPa）
            fracture_strain = 0.18  # 破断ひずみ（18%伸び）
    
            strain_elastic = np.linspace(0, yield_strain, 100)
            stress_elastic = E * strain_elastic
    
            strain_plastic = np.linspace(yield_strain, fracture_strain, 300)
            stress_plastic = yield_stress + (uts - yield_stress) * \
                            (1 - np.exp(-8 * (strain_plastic - yield_strain)))
    
            strain_necking = np.linspace(fracture_strain, fracture_strain + 0.04, 50)
            stress_necking = uts * np.exp(-15 * (strain_necking - fracture_strain))
    
            strain = np.concatenate([strain_elastic, strain_plastic, strain_necking])
            stress = np.concatenate([stress_elastic, stress_plastic, stress_necking])
    
            properties = {
                'E': E,
                'yield_stress': yield_stress,
                'UTS': uts,
                'fracture_strain': fracture_strain + 0.04,
                'type': '延性材料'
            }
    
        elif material == 'glass':
            # ガラス（脆性材料）
            E = 70e3  # ヤング率（MPa）
            fracture_stress = 70  # 破断応力（MPa）
            fracture_strain = fracture_stress / E  # 破断ひずみ（約0.1%）
    
            # 弾性領域のみ（破断まで線形）
            strain = np.linspace(0, fracture_strain, 200)
            stress = E * strain
    
            properties = {
                'E': E,
                'yield_stress': None,  # 降伏なし
                'UTS': fracture_stress,
                'fracture_strain': fracture_strain,
                'type': '脆性材料'
            }
    
        else:
            raise ValueError("material は 'steel', 'aluminum', 'glass' のいずれか")
    
        return strain, stress, properties
    
    
    # プロット作成
    fig, ax = plt.subplots(figsize=(12, 7))
    
    materials = ['steel', 'aluminum', 'glass']
    colors = ['#1f77b4', '#ff7f0e', '#d62728']
    labels = ['鋼（Steel）', 'アルミニウム（Aluminum）', 'ガラス（Glass）']
    
    for material, color, label in zip(materials, colors, labels):
        # 応力-ひずみ曲線を計算
        strain, stress, props = stress_strain_curve(material)
    
        # ひずみをパーセント表示に変換
        strain_percent = strain * 100
    
        # プロット
        ax.plot(strain_percent, stress, linewidth=2.5, color=color, label=label)
    
        # 降伏点をマーク（延性材料のみ）
        if props['yield_stress'] is not None:
            yield_strain = props['yield_stress'] / props['E']
            ax.plot(yield_strain * 100, props['yield_stress'],
                   'o', markersize=10, color=color,
                   markeredgecolor='black', markeredgewidth=1.5)
    
        # 引張強度をマーク
        if material != 'glass':
            # 延性材料：UTS点を見つける
            uts_idx = np.argmax(stress)
            ax.plot(strain_percent[uts_idx], stress[uts_idx],
                   's', markersize=10, color=color,
                   markeredgecolor='black', markeredgewidth=1.5)
    
    # 軸ラベルとタイトル
    ax.set_xlabel('ひずみ (% = $\\varepsilon$ × 100)', fontsize=13, fontweight='bold')
    ax.set_ylabel('応力 (MPa = $\\sigma$)', fontsize=13, fontweight='bold')
    ax.set_title('応力-ひずみ曲線の比較（延性材料 vs 脆性材料）',
                fontsize=14, fontweight='bold', pad=15)
    ax.legend(fontsize=11, loc='upper left')
    ax.grid(alpha=0.3, linestyle='--')
    ax.set_xlim(0, 30)
    ax.set_ylim(0, 450)
    
    # 注釈を追加
    ax.annotate('降伏点\n(Yield Point)', xy=(0.125, 250), xytext=(3, 350),
               arrowprops=dict(arrowstyle='->', color='black', lw=1.5),
               fontsize=10, ha='left',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.6))
    
    ax.annotate('引張強度\n(UTS)', xy=(12, 400), xytext=(15, 320),
               arrowprops=dict(arrowstyle='->', color='black', lw=1.5),
               fontsize=10, ha='left',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.6))
    
    ax.annotate('脆性破壊\n(ほぼ伸びなし)', xy=(0.1, 70), xytext=(2, 150),
               arrowprops=dict(arrowstyle='->', color='black', lw=1.5),
               fontsize=10, ha='left',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='lightcoral', alpha=0.6))
    
    plt.tight_layout()
    plt.show()
    
    # 材料特性の表示
    print("="*70)
    print("材料の機械的性質比較")
    print("="*70)
    
    for material, label in zip(materials, labels):
        _, _, props = stress_strain_curve(material)
        print(f"\n【{label}】")
        print(f"材料タイプ: {props['type']}")
        print(f"ヤング率 E = {props['E']/1e3:.1f} GPa")
        if props['yield_stress'] is not None:
            print(f"降伏強度 σ_y = {props['yield_stress']:.1f} MPa")
        print(f"引張強度 UTS = {props['UTS']:.1f} MPa")
        print(f"破断ひずみ ε_f = {props['fracture_strain']*100:.2f} %")
    
    print("\n" + "="*70)
    print("応力-ひずみ曲線から分かること:")
    print("- 弾性領域の傾き → ヤング率（材料の硬さ）")
    print("- 降伏点 → 塑性変形が始まる応力（設計強度の基準）")
    print("- 引張強度 → 材料が耐えられる最大応力")
    print("- 破断ひずみ → 延性の指標（大きいほど延性材料）")
    print("- 曲線下の面積 → 破壊までに吸収されるエネルギー（靭性）")
    

**解説** : 応力-ひずみ曲線は材料の機械的性質を表す最も重要なグラフです。延性材料（鋼、アルミニウム）は降伏後に塑性変形し、大きく伸びてから破断します。脆性材料（ガラス）は降伏点がなく、ほとんど伸びずに破断します。

* * *

### 硬度（Hardness）

**硬度** は、材料の表面に圧子を押し込んだときの抵抗力で、材料の硬さを示す指標です。

**主要な硬度測定法** ：

  1. **ビッカース硬度（Vickers Hardness, HV）**
     * ダイヤモンド四角錐圧子を使用
     * 荷重範囲: 1gf〜120kgf
     * あらゆる材料に適用可能
     * $HV = 1.854 \times \frac{F}{d^2}$ （F: 荷重[kgf], d: くぼみ対角線長さ[mm]）
  2. **ブリネル硬度（Brinell Hardness, HB）**
     * 超硬球（タングステンカーバイド球）を使用
     * 大きな荷重（500〜3000kgf）
     * 鋳物など粗大組織の材料に適用
     * $HB = \frac{2F}{\pi D(D - \sqrt{D^2 - d^2})}$
  3. **ロックウェル硬度（Rockwell Hardness, HR）**
     * 圧子のくい込み深さから測定
     * 迅速測定が可能
     * スケール多数（HRA, HRB, HRC など）
     * HRC: 鋼の焼入れ材に使用

材料 | ビッカース硬度（HV） | ブリネル硬度（HB） | ロックウェル硬度（HRC）  
---|---|---|---  
**軟鋼** | 120-140 | 120-140 | -  
**焼入鋼** | 600-800 | - | 55-65  
**ステンレス鋼** | 150-200 | 150-200 | 20-30  
**アルミニウム** | 20-30 | 20-30 | -  
**銅** | 40-60 | 40-60 | -  
**超硬合金** | 1400-1800 | - | -  
**ダイヤモンド** | 10000 | - | -  
  
### コード例2: 機械的性質計算機（ヤング率、降伏強度、引張強度）

引張試験データから機械的性質を計算するツールです。
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    class MechanicalPropertyCalculator:
        """
        引張試験データから機械的性質を計算するクラス
        """
    
        def __init__(self, force_data, length_data, original_length, cross_section_area):
            """
            Parameters:
            force_data: 荷重データ（N）の配列
            length_data: 長さデータ（mm）の配列
            original_length: 初期標点間距離（mm）
            cross_section_area: 断面積（mm²）
            """
            self.force = np.array(force_data)  # N
            self.length = np.array(length_data)  # mm
            self.L0 = original_length  # mm
            self.A0 = cross_section_area  # mm²
    
            # 応力とひずみを計算
            self.stress = self.force / self.A0  # MPa（N/mm² = MPa）
            self.strain = (self.length - self.L0) / self.L0  # 無次元
    
        def calculate_youngs_modulus(self, elastic_range=(0, 0.002)):
            """
            ヤング率を計算（弾性領域の傾き）
    
            Parameters:
            elastic_range: 弾性領域のひずみ範囲（タプル）
    
            Returns:
            E: ヤング率（MPa）
            """
            # 弾性領域のデータを抽出
            mask = (self.strain >= elastic_range[0]) & (self.strain <= elastic_range[1])
            strain_elastic = self.strain[mask]
            stress_elastic = self.stress[mask]
    
            # 線形フィッティング（最小二乗法）
            # stress = E * strain の傾きEを求める
            E = np.polyfit(strain_elastic, stress_elastic, 1)[0]
    
            return E
    
        def calculate_yield_strength(self, offset=0.002):
            """
            0.2%耐力（降伏強度）を計算
    
            Parameters:
            offset: オフセットひずみ（デフォルト0.2% = 0.002）
    
            Returns:
            yield_strength: 降伏強度（MPa）
            """
            # ヤング率を計算
            E = self.calculate_youngs_modulus()
    
            # オフセット線を作成（ひずみoffsetだけ平行移動）
            offset_stress = E * (self.strain - offset)
    
            # 応力-ひずみ曲線とオフセット線の交点を探す
            # 差が最小になる点を降伏点とする
            diff = np.abs(self.stress - offset_stress)
            yield_idx = np.argmin(diff[self.strain > offset])
    
            # offsetより後の範囲でインデックスを調整
            yield_idx = np.where(self.strain > offset)[0][yield_idx]
            yield_strength = self.stress[yield_idx]
    
            return yield_strength
    
        def calculate_ultimate_tensile_strength(self):
            """
            引張強度（最大応力）を計算
    
            Returns:
            UTS: 引張強度（MPa）
            """
            UTS = np.max(self.stress)
            return UTS
    
        def calculate_elongation(self):
            """
            破断伸び率を計算
    
            Returns:
            elongation: 破断伸び率（%）
            """
            max_strain = np.max(self.strain)
            elongation = max_strain * 100  # パーセント表示
            return elongation
    
        def plot_results(self):
            """
            応力-ひずみ曲線と計算結果をプロット
            """
            fig, ax = plt.subplots(figsize=(12, 7))
    
            # 応力-ひずみ曲線をプロット
            ax.plot(self.strain * 100, self.stress, 'b-', linewidth=2, label='実験データ')
    
            # ヤング率の線（弾性領域）
            E = self.calculate_youngs_modulus()
            elastic_strain = np.linspace(0, 0.002, 50)
            elastic_stress = E * elastic_strain
            ax.plot(elastic_strain * 100, elastic_stress, 'r--', linewidth=2,
                   label=f'ヤング率 E = {E/1e3:.1f} GPa')
    
            # 降伏強度をマーク
            yield_strength = self.calculate_yield_strength()
            yield_idx = np.argmin(np.abs(self.stress - yield_strength))
            ax.plot(self.strain[yield_idx] * 100, yield_strength, 'go',
                   markersize=12, label=f'降伏強度 = {yield_strength:.1f} MPa',
                   markeredgecolor='black', markeredgewidth=1.5)
    
            # 引張強度をマーク
            UTS = self.calculate_ultimate_tensile_strength()
            uts_idx = np.argmax(self.stress)
            ax.plot(self.strain[uts_idx] * 100, UTS, 'rs',
                   markersize=12, label=f'引張強度 UTS = {UTS:.1f} MPa',
                   markeredgecolor='black', markeredgewidth=1.5)
    
            # 軸ラベルとタイトル
            ax.set_xlabel('ひずみ (%)', fontsize=13, fontweight='bold')
            ax.set_ylabel('応力 (MPa)', fontsize=13, fontweight='bold')
            ax.set_title('引張試験結果と機械的性質の計算', fontsize=14, fontweight='bold')
            ax.legend(fontsize=11, loc='lower right')
            ax.grid(alpha=0.3)
    
            plt.tight_layout()
            plt.show()
    
        def print_summary(self):
            """
            計算結果のサマリーを表示
            """
            E = self.calculate_youngs_modulus()
            yield_strength = self.calculate_yield_strength()
            UTS = self.calculate_ultimate_tensile_strength()
            elongation = self.calculate_elongation()
    
            print("="*70)
            print("機械的性質の計算結果")
            print("="*70)
            print(f"\n試験片情報:")
            print(f"  初期標点間距離 L₀ = {self.L0:.2f} mm")
            print(f"  断面積 A₀ = {self.A0:.2f} mm²")
    
            print(f"\n計算された機械的性質:")
            print(f"  ヤング率 E = {E/1e3:.2f} GPa ({E:.0f} MPa)")
            print(f"  降伏強度 σ_y = {yield_strength:.2f} MPa")
            print(f"  引張強度 UTS = {UTS:.2f} MPa")
            print(f"  破断伸び率 = {elongation:.2f} %")
    
            # 材料分類の推定
            if elongation > 15:
                material_type = "延性材料（銅、アルミニウムなど）"
            elif elongation > 5:
                material_type = "中程度の延性材料（鋼など）"
            else:
                material_type = "脆性材料（ガラス、セラミックスなど）"
    
            print(f"\n推定される材料タイプ: {material_type}")
    
    
    # 実際の引張試験データをシミュレート（鋼材の例）
    # 実際には実験から得られるデータを使用
    np.random.seed(42)
    
    # シミュレーションパラメータ
    L0 = 50.0  # 初期標点間距離（mm）
    A0 = 78.5  # 断面積（mm²、直径10mmの円形断面）
    E_actual = 200e3  # 実際のヤング率（MPa = 200 GPa）
    yield_stress_actual = 250  # 実際の降伏強度（MPa）
    
    # ひずみデータ（0%から破断まで）
    strain_data = np.concatenate([
        np.linspace(0, 0.002, 50),  # 弾性領域（0〜0.2%）
        np.linspace(0.002, 0.20, 200)  # 塑性領域（0.2%〜20%）
    ])
    
    # 応力データを計算（応力-ひずみ関係）
    stress_data = np.zeros_like(strain_data)
    for i, strain in enumerate(strain_data):
        if strain <= 0.00125:  # 弾性領域
            stress_data[i] = E_actual * strain
        else:  # 塑性領域（加工硬化を考慮）
            yield_strain = yield_stress_actual / E_actual
            stress_data[i] = yield_stress_actual + \
                            (400 - yield_stress_actual) * (1 - np.exp(-8 * (strain - yield_strain)))
    
    # ノイズを追加（実際の測定を模擬）
    stress_data += np.random.normal(0, 2, len(stress_data))
    
    # 荷重データを計算（応力 = 荷重 / 断面積）
    force_data = stress_data * A0  # N
    
    # 長さデータを計算（ひずみ = (L - L0) / L0）
    length_data = L0 * (1 + strain_data)  # mm
    
    # 計算機を初期化
    calc = MechanicalPropertyCalculator(force_data, length_data, L0, A0)
    
    # 結果を表示
    calc.print_summary()
    calc.plot_results()
    
    print("\n" + "="*70)
    print("機械的性質の重要性:")
    print("- 設計: 降伏強度以下で使用（安全率を考慮）")
    print("- 材料選択: 強度と延性のバランス")
    print("- 品質管理: 引張試験で材料品質を確認")
    

**解説** : 引張試験データから、ヤング率、降伏強度、引張強度、破断伸び率を計算できます。これらの値は材料の設計や選択に不可欠な情報です。

### コード例3: 硬度換算ツール（Vickers ↔ Brinell ↔ Rockwell）

異なる硬度スケール間の換算を行うツールです。
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    class HardnessConverter:
        """
        異なる硬度スケール間の換算を行うクラス
        近似式を使用（材料によって誤差あり）
        """
    
        @staticmethod
        def vickers_to_brinell(HV):
            """
            ビッカース硬度からブリネル硬度への換算
    
            Parameters:
            HV: ビッカース硬度
    
            Returns:
            HB: ブリネル硬度
            """
            # 経験式（HV ≈ HB の範囲で使用可能）
            # 厳密にはHV = 1.05 * HB程度の関係
            HB = HV / 1.05
            return HB
    
        @staticmethod
        def vickers_to_rockwell_c(HV):
            """
            ビッカース硬度からロックウェルC硬度への換算
    
            Parameters:
            HV: ビッカース硬度
    
            Returns:
            HRC: ロックウェルC硬度
            """
            # 経験式（鋼材の場合の近似）
            # HV > 200の範囲で有効
            if HV < 200:
                return None  # 適用範囲外
    
            # 近似式: HRC = a * log(HV) + b
            # 実際のデータから導出された経験式
            HRC = 68.5 - 1000 / HV
    
            # HRCの範囲制限（20-70）
            HRC = np.clip(HRC, 20, 70)
    
            return HRC
    
        @staticmethod
        def brinell_to_vickers(HB):
            """
            ブリネル硬度からビッカース硬度への換算
    
            Parameters:
            HB: ブリネル硬度
    
            Returns:
            HV: ビッカース硬度
            """
            HV = HB * 1.05
            return HV
    
        @staticmethod
        def rockwell_c_to_vickers(HRC):
            """
            ロックウェルC硬度からビッカース硬度への換算
    
            Parameters:
            HRC: ロックウェルC硬度
    
            Returns:
            HV: ビッカース硬度
            """
            # 逆算（近似）
            HV = 1000 / (68.5 - HRC)
            return HV
    
        @staticmethod
        def estimate_tensile_strength(HV):
            """
            ビッカース硬度から引張強度を推定
    
            Parameters:
            HV: ビッカース硬度
    
            Returns:
            UTS: 推定引張強度（MPa）
            """
            # 経験式（鋼材）: UTS ≈ 3.3 * HV
            UTS = 3.3 * HV
            return UTS
    
    
    # 換算ツールの使用例
    converter = HardnessConverter()
    
    print("="*70)
    print("硬度換算ツール")
    print("="*70)
    
    # テストデータ（いくつかの材料）
    materials = [
        {'name': '軟鋼', 'HV': 130},
        {'name': 'ステンレス鋼', 'HV': 180},
        {'name': '焼入鋼（低温焼戻し）', 'HV': 600},
        {'name': '焼入鋼（高温焼戻し）', 'HV': 400},
        {'name': '工具鋼', 'HV': 750},
    ]
    
    print("\n硬度換算表:")
    print("-" * 70)
    print(f"{'材料':<20} {'HV':>8} {'HB':>8} {'HRC':>8} {'推定UTS(MPa)':>15}")
    print("-" * 70)
    
    for mat in materials:
        HV = mat['HV']
        HB = converter.vickers_to_brinell(HV)
        HRC = converter.vickers_to_rockwell_c(HV)
        UTS = converter.estimate_tensile_strength(HV)
    
        if HRC is not None:
            print(f"{mat['name']:<20} {HV:>8.0f} {HB:>8.0f} {HRC:>8.1f} {UTS:>15.0f}")
        else:
            print(f"{mat['name']:<20} {HV:>8.0f} {HB:>8.0f} {'N/A':>8} {UTS:>15.0f}")
    
    # 硬度と引張強度の関係をプロット
    HV_range = np.linspace(100, 800, 100)
    UTS_range = converter.estimate_tensile_strength(HV_range)
    HB_range = converter.vickers_to_brinell(HV_range)
    HRC_range = np.array([converter.vickers_to_rockwell_c(hv) for hv in HV_range])
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 硬度スケール間の換算関係
    ax1 = axes[0]
    ax1.plot(HV_range, HV_range, 'b-', linewidth=2, label='HV')
    ax1.plot(HV_range, HB_range, 'r--', linewidth=2, label='HB')
    ax1.set_xlabel('ビッカース硬度 HV', fontsize=12, fontweight='bold')
    ax1.set_ylabel('硬度値', fontsize=12, fontweight='bold')
    ax1.set_title('硬度スケールの換算（HV ↔ HB）', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(alpha=0.3)
    
    # 硬度と引張強度の関係
    ax2 = axes[1]
    ax2.plot(HV_range, UTS_range, 'g-', linewidth=2.5)
    ax2.set_xlabel('ビッカース硬度 HV', fontsize=12, fontweight='bold')
    ax2.set_ylabel('推定引張強度 (MPa)', fontsize=12, fontweight='bold')
    ax2.set_title('硬度から引張強度を推定\n（経験式: UTS ≈ 3.3 × HV）',
                 fontsize=13, fontweight='bold')
    ax2.grid(alpha=0.3)
    
    # データ点をプロット
    for mat in materials:
        HV = mat['HV']
        UTS = converter.estimate_tensile_strength(HV)
        ax2.plot(HV, UTS, 'ro', markersize=10, markeredgecolor='black', markeredgewidth=1.5)
    
    plt.tight_layout()
    plt.show()
    
    # 硬度測定法の比較表を出力
    print("\n" + "="*70)
    print("硬度測定法の比較")
    print("="*70)
    print("\n【ビッカース硬度（HV）】")
    print("- 圧子: ダイヤモンド四角錐（対面角136°）")
    print("- 荷重: 1gf〜120kgf（試験力により表記: HV0.1, HV10など）")
    print("- 特徴: あらゆる材料に適用可能、測定範囲が広い")
    print("- 用途: 薄板、表面処理層、小型部品")
    
    print("\n【ブリネル硬度（HB）】")
    print("- 圧子: 超硬球（直径2.5mm, 5mm, 10mm）")
    print("- 荷重: 500〜3000kgf")
    print("- 特徴: くぼみが大きく、粗大組織でも測定可能")
    print("- 用途: 鋳物、大型部品、粗大組織材料")
    
    print("\n【ロックウェル硬度（HRC）】")
    print("- 圧子: ダイヤモンド円錐（HRC）、鋼球（HRB）")
    print("- 荷重: 60kgf（HRA）、100kgf（HRB）、150kgf（HRC）")
    print("- 特徴: くい込み深さで測定、迅速測定可能")
    print("- 用途: 鋼の焼入れ材（HRC）、軟鋼・非鉄金属（HRB）")
    
    print("\n" + "="*70)
    print("硬度と引張強度の関係:")
    print("- 経験則: UTS (MPa) ≈ 3.3 × HV（鋼材の場合）")
    print("- 硬度から強度を簡易的に推定可能（非破壊）")
    print("- 注意: 材料種類により係数が異なる（3.0〜3.5程度）")
    

**解説** : 硬度測定法は複数あり、測定目的や材料に応じて使い分けます。ビッカース硬度は最も汎用性が高く、ブリネル硬度は大型部品、ロックウェル硬度は迅速測定に適しています。硬度から引張強度を推定することも可能です。

* * *

## 4.2 電気的性質（導電性、半導性、絶縁性）

### 電気伝導度と抵抗率

**電気伝導度（Electrical Conductivity, σ）** は、電流の流れやすさを表します：

$$\sigma = \frac{1}{\rho}$$

ここで、$\rho$は**抵抗率（Resistivity）** （単位: Ω·m）です。

**材料の分類** ：

  * **導体（Conductor）** : $\rho < 10^{-5}$ Ω·m（金属）
  * **半導体（Semiconductor）** : $10^{-5} < \rho < 10^{7}$ Ω·m（Si, Ge）
  * **絶縁体（Insulator）** : $\rho > 10^{7}$ Ω·m（ガラス、セラミックス、高分子）

材料 | 抵抗率（Ω·m, 20°C） | 分類  
---|---|---  
**銀（Ag）** | 1.59 × 10⁻⁸ | 導体  
**銅（Cu）** | 1.68 × 10⁻⁸ | 導体  
**金（Au）** | 2.44 × 10⁻⁸ | 導体  
**アルミニウム（Al）** | 2.82 × 10⁻⁸ | 導体  
**ゲルマニウム（Ge）** | 4.6 × 10⁻¹ | 半導体  
**シリコン（Si）** | 6.4 × 10² | 半導体  
**ガラス（SiO₂）** | 10¹⁰ - 10¹⁴ | 絶縁体  
**ポリエチレン** | 10¹⁶ | 絶縁体  
  
### バンド構造と電気伝導

**バンド理論** では、材料の電子状態をエネルギーバンドで表します：

  * **価電子帯（Valence Band）** : 電子で満たされたバンド
  * **伝導帯（Conduction Band）** : 空のバンド（電子が自由に動ける）
  * **バンドギャップ（Band Gap, Eg）** : 価電子帯と伝導帯の間のエネルギー差

**材料分類とバンドギャップ** ：

  * **金属（Conductor）** : Eg = 0（価電子帯と伝導帯が重なる）
  * **半導体（Semiconductor）** : 0 < Eg < 3 eV（室温で励起可能）
  * **絶縁体（Insulator）** : Eg > 3 eV（励起困難）

半導体材料 | バンドギャップ Eg（eV, 300K） | 用途  
---|---|---  
**Si** | 1.12 | 集積回路、太陽電池  
**Ge** | 0.66 | 赤外線検出器  
**GaAs** | 1.42 | 高速デバイス、LED  
**GaN** | 3.44 | 青色LED、パワーデバイス  
**InP** | 1.35 | 光通信デバイス  
**SiC** | 3.26 | 高温・高電圧デバイス  
  
### コード例4: バンドギャップと電気伝導度の関係可視化

バンドギャップと電気伝導度の関係を可視化します。
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    # 物理定数
    k_B = 8.617e-5  # ボルツマン定数（eV/K）
    
    def intrinsic_carrier_concentration(Eg, T=300):
        """
        真性キャリア濃度を計算
    
        Parameters:
        Eg: バンドギャップ（eV）
        T: 温度（K）
    
        Returns:
        n_i: 真性キャリア濃度（cm⁻³）
        """
        # 簡略化した近似式
        # n_i ∝ exp(-Eg / 2k_B T)
        # 実際にはn_i = sqrt(N_c * N_v) * exp(-Eg / 2k_B T)
        # ここでは相対的な値を計算
        n_i = 1e19 * np.exp(-Eg / (2 * k_B * T))
        return n_i
    
    def electrical_conductivity(n, mu=1000):
        """
        電気伝導度を計算
    
        Parameters:
        n: キャリア濃度（cm⁻³）
        mu: 移動度（cm²/V·s）
    
        Returns:
        sigma: 電気伝導度（S/cm）
        """
        q = 1.602e-19  # 電荷素量（C）
        sigma = q * n * mu  # S/cm
        return sigma
    
    # バンドギャップ範囲（0〜5 eV）
    Eg_range = np.linspace(0.1, 5, 100)
    
    # 各バンドギャップに対するキャリア濃度と電気伝導度を計算
    n_i_range = np.array([intrinsic_carrier_concentration(Eg) for Eg in Eg_range])
    sigma_range = electrical_conductivity(n_i_range)
    
    # 抵抗率を計算
    rho_range = 1 / sigma_range  # Ω·cm
    
    # プロット作成
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # バンドギャップ vs キャリア濃度
    ax1 = axes[0]
    ax1.semilogy(Eg_range, n_i_range, 'b-', linewidth=2.5)
    ax1.set_xlabel('バンドギャップ $E_g$ (eV)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('真性キャリア濃度 $n_i$ (cm⁻³)', fontsize=12, fontweight='bold')
    ax1.set_title('バンドギャップとキャリア濃度の関係', fontsize=13, fontweight='bold')
    ax1.grid(alpha=0.3, which='both')
    
    # 材料をプロット
    materials_bandgap = [
        ('Ge', 0.66), ('Si', 1.12), ('GaAs', 1.42),
        ('InP', 1.35), ('SiC', 3.26), ('GaN', 3.44)
    ]
    for name, Eg in materials_bandgap:
        n_i = intrinsic_carrier_concentration(Eg)
        ax1.plot(Eg, n_i, 'ro', markersize=10, markeredgecolor='black', markeredgewidth=1.5)
        ax1.annotate(name, xy=(Eg, n_i), xytext=(Eg+0.1, n_i*2),
                    fontsize=9, ha='left')
    
    # バンドギャップ vs 抵抗率
    ax2 = axes[1]
    ax2.semilogy(Eg_range, rho_range, 'g-', linewidth=2.5)
    ax2.set_xlabel('バンドギャップ $E_g$ (eV)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('抵抗率 $\\rho$ (Ω·cm)', fontsize=12, fontweight='bold')
    ax2.set_title('バンドギャップと抵抗率の関係', fontsize=13, fontweight='bold')
    ax2.grid(alpha=0.3, which='both')
    
    # 材料分類の領域を色分け
    ax2.axhspan(1e-8, 1e-5, alpha=0.2, color='blue', label='導体領域')
    ax2.axhspan(1e-5, 1e7, alpha=0.2, color='yellow', label='半導体領域')
    ax2.axhspan(1e7, 1e20, alpha=0.2, color='red', label='絶縁体領域')
    ax2.legend(fontsize=10, loc='upper left')
    
    # 材料をプロット
    for name, Eg in materials_bandgap:
        n_i = intrinsic_carrier_concentration(Eg)
        sigma = electrical_conductivity(n_i)
        rho = 1 / sigma
        ax2.plot(Eg, rho, 'ro', markersize=10, markeredgecolor='black', markeredgewidth=1.5)
    
    plt.tight_layout()
    plt.show()
    
    # 材料特性の表示
    print("="*70)
    print("半導体材料のバンドギャップと電気的性質")
    print("="*70)
    
    print(f"\n{'材料':<10} {'Eg(eV)':>10} {'n_i(cm⁻³)':>15} {'ρ(Ω·cm)':>15} {'分類':<10}")
    print("-" * 70)
    
    for name, Eg in materials_bandgap:
        n_i = intrinsic_carrier_concentration(Eg)
        sigma = electrical_conductivity(n_i)
        rho = 1 / sigma
    
        if rho < 1e-5:
            classification = "導体"
        elif rho < 1e7:
            classification = "半導体"
        else:
            classification = "絶縁体"
    
        print(f"{name:<10} {Eg:>10.2f} {n_i:>15.2e} {rho:>15.2e} {classification:<10}")
    
    print("\n" + "="*70)
    print("バンドギャップの重要性:")
    print("- Eg が小さい → キャリア濃度が高い → 電気伝導度が高い")
    print("- Eg が大きい → キャリア濃度が低い → 絶縁性が高い")
    print("- Si（Eg=1.12eV）: 最も重要な半導体材料（室温で適度な伝導性）")
    print("- GaN（Eg=3.44eV）: ワイドバンドギャップ半導体（高温・高電圧動作）")
    

**解説** : バンドギャップが小さいほど、室温でのキャリア濃度が高くなり、電気伝導度が大きくなります（抵抗率が小さくなります）。半導体のバンドギャップは材料の用途を決定する重要なパラメータです。

### コード例5: 抵抗率の温度依存性プロット（金属 vs 半導体）

金属と半導体では、抵抗率の温度依存性が逆になることを可視化します。
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    def resistivity_metal(T, rho_0=1.68e-8, alpha=0.0039):
        """
        金属の抵抗率（温度依存性）
    
        Parameters:
        T: 温度（K）
        rho_0: 基準温度（273K）での抵抗率（Ω·m）
        alpha: 温度係数（1/K）
    
        Returns:
        rho: 抵抗率（Ω·m）
        """
        T_0 = 273  # 基準温度（K）
        rho = rho_0 * (1 + alpha * (T - T_0))
        return rho
    
    def resistivity_semiconductor(T, Eg=1.12, rho_room=640):
        """
        半導体の抵抗率（温度依存性）
    
        Parameters:
        T: 温度（K）
        Eg: バンドギャップ（eV）
        rho_room: 室温（300K）での抵抗率（Ω·m）
    
        Returns:
        rho: 抵抗率（Ω·m）
        """
        k_B = 8.617e-5  # ボルツマン定数（eV/K）
        T_room = 300  # 室温（K）
    
        # 真性半導体の抵抗率は exp(Eg / 2k_B T) に比例
        rho = rho_room * np.exp(Eg / (2 * k_B) * (1/T - 1/T_room))
        return rho
    
    # 温度範囲（200K 〜 500K）
    T_range = np.linspace(200, 500, 100)
    
    # 金属（銅）の抵抗率
    rho_metal = resistivity_metal(T_range, rho_0=1.68e-8, alpha=0.0039)
    
    # 半導体（シリコン）の抵抗率
    rho_si = resistivity_semiconductor(T_range, Eg=1.12, rho_room=640)
    
    # プロット作成
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 通常スケール
    ax1 = axes[0]
    ax1.plot(T_range - 273, rho_metal * 1e8, 'b-', linewidth=2.5, label='金属（Cu）')
    ax1_twin = ax1.twinx()
    ax1_twin.plot(T_range - 273, rho_si, 'r--', linewidth=2.5, label='半導体（Si）')
    
    ax1.set_xlabel('温度 (°C)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('抵抗率（金属, 10⁻⁸ Ω·m）', fontsize=11, fontweight='bold', color='b')
    ax1_twin.set_ylabel('抵抗率（半導体, Ω·m）', fontsize=11, fontweight='bold', color='r')
    ax1.set_title('抵抗率の温度依存性（通常スケール）', fontsize=13, fontweight='bold')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1_twin.tick_params(axis='y', labelcolor='r')
    ax1.grid(alpha=0.3)
    
    # 凡例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=10, loc='upper left')
    
    # 対数スケール
    ax2 = axes[1]
    ax2.semilogy(T_range - 273, rho_metal, 'b-', linewidth=2.5, label='金属（Cu）')
    ax2.semilogy(T_range - 273, rho_si, 'r--', linewidth=2.5, label='半導体（Si）')
    ax2.set_xlabel('温度 (°C)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('抵抗率 (Ω·m)', fontsize=12, fontweight='bold')
    ax2.set_title('抵抗率の温度依存性（対数スケール）', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(alpha=0.3, which='both')
    
    plt.tight_layout()
    plt.show()
    
    # 温度係数の計算と表示
    print("="*70)
    print("抵抗率の温度依存性")
    print("="*70)
    
    # 特定温度での値を計算
    temps_celsius = [0, 25, 100, 200]
    temps_kelvin = [t + 273 for t in temps_celsius]
    
    print("\n【金属（銅）の抵抗率】")
    print(f"{'温度(°C)':>10} {'抵抗率(10⁻⁸ Ω·m)':>25} {'変化率(%)':>15}")
    print("-" * 70)
    rho_ref = resistivity_metal(273)
    for T_c, T_k in zip(temps_celsius, temps_kelvin):
        rho = resistivity_metal(T_k)
        change = ((rho - rho_ref) / rho_ref) * 100
        print(f"{T_c:>10} {rho*1e8:>25.4f} {change:>15.2f}")
    
    print("\n【半導体（シリコン）の抵抗率】")
    print(f"{'温度(°C)':>10} {'抵抗率(Ω·m)':>25} {'変化率(%)':>15}")
    print("-" * 70)
    rho_ref = resistivity_semiconductor(300)
    for T_c, T_k in zip(temps_celsius, temps_kelvin):
        rho = resistivity_semiconductor(T_k)
        change = ((rho - rho_ref) / rho_ref) * 100
        print(f"{T_c:>10} {rho:>25.2e} {change:>15.2f}")
    
    print("\n" + "="*70)
    print("温度依存性の違い:")
    print("\n【金属】")
    print("- 温度上昇 → 抵抗率増加（正の温度係数）")
    print("- 理由: 格子振動が増大し、電子散乱が増える")
    print("- 温度係数 α ≈ +0.4% / K（銅の場合）")
    print("- 応用: 測温抵抗体（白金測温抵抗体など）")
    
    print("\n【半導体】")
    print("- 温度上昇 → 抵抗率減少（負の温度係数）")
    print("- 理由: 熱励起によりキャリア濃度が増加")
    print("- 温度係数は負で大きい（-数%/K）")
    print("- 応用: サーミスタ（温度センサ）")
    

**解説** : 金属は温度上昇で抵抗率が増加し、半導体は温度上昇で抵抗率が減少します。これは電気伝導のメカニズムの違いによります。金属では格子振動による散乱が支配的で、半導体では熱励起によるキャリア濃度の増加が支配的です。

* * *

## 4.3 熱的性質（熱伝導、熱膨張）

### 熱伝導率

**熱伝導率（Thermal Conductivity, κ）** は、熱の伝わりやすさを表します：

$$q = -\kappa \nabla T$$

ここで、$q$は熱流束（W/m²）、$\nabla T$は温度勾配（K/m）です。

**材料分類** ：

  * **金属** : κ = 50-400 W/(m·K)（高い熱伝導率）
  * **セラミックス** : κ = 1-50 W/(m·K)
  * **高分子** : κ = 0.1-0.5 W/(m·K)（低い熱伝導率）

### 線膨張係数

**線膨張係数（Coefficient of Thermal Expansion, CTE, α）** は、温度変化に対する長さの変化率です：

$$\alpha = \frac{1}{L} \frac{dL}{dT}$$

単位は 1/K または ppm/K（10⁻⁶/K）です。

### 比熱

**比熱（Specific Heat Capacity, c）** は、単位質量の物質を1K温度上昇させるのに必要な熱量です：

$$Q = mc\Delta T$$

単位は J/(kg·K) です。

### コード例6: 熱的性質の比較（熱伝導率、線膨張係数、比熱）

代表的な材料の熱的性質を比較します。
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    # 材料の熱的性質データベース
    materials_thermal = {
        # 金属
        '銅（Cu）': {
            'thermal_conductivity': 401,  # W/(m·K)
            'thermal_expansion': 16.5,    # ppm/K (10⁻⁶/K)
            'specific_heat': 385,         # J/(kg·K)
            'density': 8960,              # kg/m³
            'category': '金属'
        },
        'アルミニウム（Al）': {
            'thermal_conductivity': 237,
            'thermal_expansion': 23.1,
            'specific_heat': 897,
            'density': 2700,
            'category': '金属'
        },
        '鉄（Fe）': {
            'thermal_conductivity': 80,
            'thermal_expansion': 11.8,
            'specific_heat': 449,
            'density': 7874,
            'category': '金属'
        },
        'ステンレス鋼（SUS304）': {
            'thermal_conductivity': 16,
            'thermal_expansion': 17.3,
            'specific_heat': 500,
            'density': 8000,
            'category': '金属'
        },
        # セラミックス
        'アルミナ（Al₂O₃）': {
            'thermal_conductivity': 30,
            'thermal_expansion': 8.1,
            'specific_heat': 775,
            'density': 3950,
            'category': 'セラミックス'
        },
        '窒化ケイ素（Si₃N₄）': {
            'thermal_conductivity': 28,
            'thermal_expansion': 3.2,
            'specific_heat': 680,
            'density': 3200,
            'category': 'セラミックス'
        },
        'ガラス（SiO₂）': {
            'thermal_conductivity': 1.4,
            'thermal_expansion': 0.55,
            'specific_heat': 750,
            'density': 2200,
            'category': 'セラミックス'
        },
        # 高分子
        'ポリエチレン（PE）': {
            'thermal_conductivity': 0.42,
            'thermal_expansion': 100,
            'specific_heat': 2300,
            'density': 950,
            'category': '高分子'
        },
        'ポリスチレン（PS）': {
            'thermal_conductivity': 0.13,
            'thermal_expansion': 70,
            'specific_heat': 1300,
            'density': 1050,
            'category': '高分子'
        }
    }
    
    # データを整理
    materials = list(materials_thermal.keys())
    thermal_cond = [materials_thermal[m]['thermal_conductivity'] for m in materials]
    thermal_exp = [materials_thermal[m]['thermal_expansion'] for m in materials]
    specific_heat = [materials_thermal[m]['specific_heat'] for m in materials]
    categories = [materials_thermal[m]['category'] for m in materials]
    
    # カテゴリごとに色分け
    color_map = {'金属': '#1f77b4', 'セラミックス': '#ff7f0e', '高分子': '#2ca02c'}
    colors = [color_map[cat] for cat in categories]
    
    # プロット作成
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 熱伝導率の比較
    ax1 = axes[0, 0]
    y_pos = np.arange(len(materials))
    bars1 = ax1.barh(y_pos, thermal_cond, color=colors, edgecolor='black', linewidth=1.5, alpha=0.7)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(materials, fontsize=9)
    ax1.set_xlabel('熱伝導率 κ (W/(m·K))', fontsize=11, fontweight='bold')
    ax1.set_title('熱伝導率の比較', fontsize=12, fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)
    ax1.set_xscale('log')
    
    # 値をバーの端に表示
    for i, (bar, val) in enumerate(zip(bars1, thermal_cond)):
        ax1.text(val * 1.1, bar.get_y() + bar.get_height()/2,
                f'{val:.1f}', va='center', fontsize=8)
    
    # 線膨張係数の比較
    ax2 = axes[0, 1]
    bars2 = ax2.barh(y_pos, thermal_exp, color=colors, edgecolor='black', linewidth=1.5, alpha=0.7)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(materials, fontsize=9)
    ax2.set_xlabel('線膨張係数 α (ppm/K = 10⁻⁶/K)', fontsize=11, fontweight='bold')
    ax2.set_title('線膨張係数の比較', fontsize=12, fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)
    ax2.set_xscale('log')
    
    for i, (bar, val) in enumerate(zip(bars2, thermal_exp)):
        ax2.text(val * 1.1, bar.get_y() + bar.get_height()/2,
                f'{val:.1f}', va='center', fontsize=8)
    
    # 比熱の比較
    ax3 = axes[1, 0]
    bars3 = ax3.barh(y_pos, specific_heat, color=colors, edgecolor='black', linewidth=1.5, alpha=0.7)
    ax3.set_yticks(y_pos)
    ax3.set_yticklabels(materials, fontsize=9)
    ax3.set_xlabel('比熱 c (J/(kg·K))', fontsize=11, fontweight='bold')
    ax3.set_title('比熱の比較', fontsize=12, fontweight='bold')
    ax3.grid(axis='x', alpha=0.3)
    
    for i, (bar, val) in enumerate(zip(bars3, specific_heat)):
        ax3.text(val + 50, bar.get_y() + bar.get_height()/2,
                f'{val:.0f}', va='center', fontsize=8)
    
    # 熱伝導率 vs 線膨張係数の散布図
    ax4 = axes[1, 1]
    for cat in ['金属', 'セラミックス', '高分子']:
        indices = [i for i, c in enumerate(categories) if c == cat]
        tc = [thermal_cond[i] for i in indices]
        te = [thermal_exp[i] for i in indices]
        ax4.scatter(tc, te, s=150, c=color_map[cat], label=cat,
                   edgecolors='black', linewidth=1.5, alpha=0.7)
    
    ax4.set_xlabel('熱伝導率 κ (W/(m·K))', fontsize=11, fontweight='bold')
    ax4.set_ylabel('線膨張係数 α (ppm/K)', fontsize=11, fontweight='bold')
    ax4.set_title('熱伝導率 vs 線膨張係数', fontsize=12, fontweight='bold')
    ax4.set_xscale('log')
    ax4.set_yscale('log')
    ax4.legend(fontsize=10)
    ax4.grid(alpha=0.3, which='both')
    
    # 材料名をプロット
    for i, mat in enumerate(materials):
        ax4.annotate(mat.split('（')[0],
                    xy=(thermal_cond[i], thermal_exp[i]),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=7, alpha=0.8)
    
    plt.tight_layout()
    plt.show()
    
    # 熱拡散率の計算
    print("="*70)
    print("材料の熱的性質比較")
    print("="*70)
    
    print(f"\n{'材料':<20} {'κ(W/m·K)':>12} {'α(ppm/K)':>12} {'c(J/kg·K)':>12} {'a(mm²/s)':>12}")
    print("-" * 70)
    
    for mat in materials:
        props = materials_thermal[mat]
        kappa = props['thermal_conductivity']
        alpha_exp = props['thermal_expansion']
        c = props['specific_heat']
        rho = props['density']
    
        # 熱拡散率 a = κ / (ρ * c) の計算
        thermal_diffusivity = kappa / (rho * c) * 1e6  # mm²/s
    
        print(f"{mat:<20} {kappa:>12.2f} {alpha_exp:>12.2f} {c:>12.0f} {thermal_diffusivity:>12.3f}")
    
    print("\n" + "="*70)
    print("熱的性質の意味と応用:")
    print("="*70)
    
    print("\n【熱伝導率 κ】")
    print("- 高い → 熱を素早く伝える → ヒートシンク、放熱材料")
    print("- 低い → 断熱性が高い → 断熱材、保温材")
    print("- 金属 > セラミックス > 高分子の順")
    print("- 応用: 銅（放熱）、ステンレス（断熱）")
    
    print("\n【線膨張係数 α】")
    print("- 高い → 温度変化で大きく伸縮 → 熱応力が発生しやすい")
    print("- 低い → 寸法安定性が高い → 精密機器に適する")
    print("- 高分子 > 金属 > セラミックスの順")
    print("- 応用: ガラス（低CTE）、異材接合時の熱応力管理")
    
    print("\n【比熱 c】")
    print("- 高い → 温度変化しにくい → 蓄熱材料")
    print("- 低い → 素早く温度が変わる → 熱応答が速い")
    print("- 高分子 > セラミックス > 金属の順（質量あたり）")
    print("- 応用: 水（高比熱、冷却材）、金属（低比熱、調理器具）")
    
    print("\n【熱拡散率 a = κ/(ρc)】")
    print("- 熱が材料中を拡散する速さ")
    print("- 高い → 全体が素早く均一温度になる")
    print("- 金属が最も高い（銅、アルミニウム）")
    

**解説** : 材料の熱的性質は、熱管理設計に重要です。金属は熱伝導率が高く放熱に適し、高分子は熱伝導率が低く断熱に適します。線膨張係数は異材接合時の熱応力を考慮する際に重要です。

* * *

## 4.4 光学的性質（透明性、色）

### 透明性と不透明性

**透明（Transparent）** : 可視光がほぼ吸収されずに透過

  * 例: ガラス、透明高分子（PMMA, PC）
  * 条件: バンドギャップ > 可視光のエネルギー（約1.8〜3.1 eV）

**半透明（Translucent）** : 光が散乱されながら透過

  * 例: すりガラス、薄い紙

**不透明（Opaque）** : 光が吸収または反射される

  * 例: 金属、黒色材料
  * 金属: 自由電子による反射

### 色と吸収スペクトル

材料の**色** は、可視光の特定波長を吸収し、残りを反射・透過することで生じます。

**可視光の波長範囲** :

  * 紫: 380-450 nm
  * 青: 450-495 nm
  * 緑: 495-570 nm
  * 黄: 570-590 nm
  * 橙: 590-620 nm
  * 赤: 620-750 nm

**補色の関係** : ある色を吸収すると、その補色が見える

  * 青を吸収 → 橙色に見える
  * 赤を吸収 → 青緑色に見える

### 屈折率

**屈折率（Refractive Index, n）** は、光が物質中を進む速さの比です：

$$n = \frac{c}{v}$$

ここで、$c$は真空中の光速、$v$は物質中の光速です。

材料 | 屈折率（589nm, D線） | 透明性  
---|---|---  
**真空** | 1.0000 | -  
**空気** | 1.0003 | -  
**水** | 1.333 | 透明  
**石英ガラス（SiO₂）** | 1.458 | 透明  
**ソーダガラス** | 1.52 | 透明  
**PMMA（アクリル）** | 1.49 | 透明  
**ポリカーボネート（PC）** | 1.586 | 透明  
**ダイヤモンド** | 2.417 | 透明  
  
**光学的性質の応用** :

  * **レンズ** : 高屈折率材料（光学ガラス、高分子）
  * **光ファイバ** : 低損失透明材料（石英ガラス）
  * **反射防止コーティング** : 薄膜干渉を利用
  * **着色材料** : 顔料・染料による吸収スペクトル制御
  * **太陽電池** : 可視光吸収材料（Si, GaAs など）

> **まとめ** : 材料の性質（機械的・電気的・熱的・光学的）は、原子構造と結晶構造に深く関連しています。用途に応じて適切な材料を選択するには、これらの性質を定量的に理解し、比較することが重要です。

* * *

## 4.5 本章のまとめ

### 学んだこと

  1. **機械的性質**
     * 応力-ひずみ曲線: ヤング率、降伏強度、引張強度、破断伸び
     * 延性材料 vs 脆性材料の違い
     * 硬度測定法（Vickers, Brinell, Rockwell）と換算
  2. **電気的性質**
     * 導体・半導体・絶縁体の分類（抵抗率による）
     * バンドギャップと電気伝導度の関係
     * 金属と半導体の温度依存性の違い（正 vs 負の温度係数）
  3. **熱的性質**
     * 熱伝導率: 金属 > セラミックス > 高分子
     * 線膨張係数: 高分子 > 金属 > セラミックス
     * 比熱と熱拡散率の意味
  4. **光学的性質**
     * 透明性の条件（バンドギャップ > 可視光エネルギー）
     * 色と吸収スペクトルの関係
     * 屈折率と光学応用

### 重要なポイント

  * 材料の性質は**構造（原子配列・結晶構造・化学結合）** に起因する
  * 機械的性質は材料選択の最も基本的な指標
  * 電気的性質はバンド構造で説明される
  * 熱的性質は熱管理設計に不可欠
  * Pythonで材料特性を定量的に計算・比較できる

### 次の章へ

第5章では、**Pythonで学ぶ結晶構造可視化** を学びます：

  * pymatgen入門（結晶構造ライブラリ）
  * CIFファイルの読み込みと構造解析
  * Materials Projectデータベースの活用
  * 代表的材料（Si, Fe, Al₂O₃）の構造解析
  * 総合ワークフロー（構造→解析→可視化→特性予測）
