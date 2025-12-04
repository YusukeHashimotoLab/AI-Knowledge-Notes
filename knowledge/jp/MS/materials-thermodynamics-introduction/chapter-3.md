---
title: "第3章: 相平衡と相図の基礎"
chapter_title: "第3章: 相平衡と相図の基礎"
subtitle: Phase Equilibria and Phase Diagrams
reading_time: 26-32分
difficulty: 中級
code_examples: 8
---

## 学習目標

この章を学ぶことで、以下のスキルを習得できます：

  * 相（phase）の定義と種類を理解し、材料中の相を識別できる
  * 平衡条件と化学ポテンシャル平衡の関係を説明できる
  * ギブスの相律（F = C - P + 2）を適用し、系の自由度を計算できる
  * 一成分系相図（圧力-温度図）を読み、解釈できる
  * クラペイロンの式とクラウジウス-クラペイロンの式を使って相境界を計算できる
  * 相転移の分類（一次、二次相転移）を理解し、実例を挙げられる
  * レバールール（てこの原理）を使って相分率を計算できる
  * Pythonで相図を描画し、実材料の相転移を予測できる

## 1\. 相（Phase）とは何か

### 1.1 相の定義

材料科学において、**相（phase）** は、物理的・化学的に均一な領域を指します。相は、明確な界面で他の相と区別されます。

#### 相の定義と特徴

**相** とは、以下の特徴を持つ物質の状態：

  * **組成が均一** : 相内のどの位置でも化学組成が同じ
  * **物性が均一** : 密度、屈折率、結晶構造などが一定
  * **明確な界面** : 異なる相の間には明確な境界が存在
  * **物理的に分離可能** : 原理的に他の相から分離できる

### 1.2 相の種類

材料には様々な相が存在します：

相の種類 | 説明 | 具体例  
---|---|---  
**気相** | 気体状態。分子間距離が大きく自由に運動 | H₂O蒸気、Ar雰囲気  
**液相** | 液体状態。分子が密集するが流動性あり | 液体水、溶融金属（Fe液相）  
**固相** | 固体状態。原子が規則的または不規則に配列 | 氷（H₂O固相）、Fe結晶  
**結晶相** | 原子が周期的に配列した固相 | α-Fe（BCC）、γ-Fe（FCC）  
**非晶質相** | 長距離秩序のない固相 | ガラス（SiO₂非晶質）、金属ガラス  
  
#### 具体例: 純鉄（Fe）の相

純鉄は温度により異なる結晶構造を持つ相が現れます：

  * **α-Fe（フェライト）** : 室温～912°C、体心立方（BCC）構造
  * **γ-Fe（オーステナイト）** : 912°C～1394°C、面心立方（FCC）構造
  * **δ-Fe** : 1394°C～1538°C（融点）、体心立方（BCC）構造
  * **液相Fe** : 1538°C以上、原子が不規則に流動

これらは**同素体（allotrope）** と呼ばれ、同じ元素でも結晶構造が異なる相です。

### 1.3 相と組織の違い

#### 注意: 相（Phase）と組織（Microstructure）は異なる概念

  * **相** : 熱力学的に定義される均一領域（α相、β相など）
  * **組織** : 相の空間的配置や形状（粒径、層状、球状など）

例: パーライト組織は、α-Fe（フェライト）とFe₃C（セメンタイト）の**2つの相** が層状に配列した**組織** です。

## 2\. 平衡状態と平衡条件

### 2.1 平衡状態の定義

前章で学んだように、一定温度・圧力下では、系は**ギブスエネルギー（G）が最小** の状態で平衡に達します。複数の相が共存する場合、平衡条件はより具体的に表されます。

#### 多相系の平衡条件

相 α、β、γ が平衡共存するためには、以下の条件が満たされる必要があります：

**1\. 温度平衡** : すべての相で温度が等しい

$$ T^\alpha = T^\beta = T^\gamma $$ 

**2\. 圧力平衡** : すべての相で圧力が等しい（界面張力が無視できる場合）

$$ P^\alpha = P^\beta = P^\gamma $$ 

**3\. 化学ポテンシャル平衡** : 各成分の化学ポテンシャルが全相で等しい

$$ \mu_i^\alpha = \mu_i^\beta = \mu_i^\gamma \quad \text{（成分 $i$ について）} $$ 

### 2.2 化学ポテンシャル平衡の物理的意味

化学ポテンシャル平衡の条件 $\mu_i^\alpha = \mu_i^\beta$ は、「成分 $i$ が α相から β相へ移動する駆動力がゼロ」であることを意味します。

#### 水の蒸発平衡での理解

コップに入った水が蒸発と凝縮を繰り返し、最終的に液相と気相が共存する状態を考えます：

  * **非平衡状態** : $\mu_{\text{H}_2\text{O}}^{\text{液}} > \mu_{\text{H}_2\text{O}}^{\text{気}}$ → 蒸発が優勢
  * **平衡状態** : $\mu_{\text{H}_2\text{O}}^{\text{液}} = \mu_{\text{H}_2\text{O}}^{\text{気}}$ → 蒸発と凝縮が釣り合う

この平衡状態での気相の圧力が**飽和蒸気圧** です。

### 2.3 平衡条件の決定フロー
    
    
    ```mermaid
    flowchart TD
        A[初期状態: 任意の温度・圧力] --> B{すべての相でT, P が等しいか?}
        B -->|No| C[熱・力学的平衡化T, P を均一にする]
        C --> B
        B -->|Yes| D{各成分iについてμ_i が全相で等しいか?}
        D -->|No| E[物質移動高μ相 → 低μ相]
        E --> D
        D -->|Yes| F[化学平衡達成]
        F --> G[系全体のギブスエネルギーが最小値に到達]
        G --> H[平衡状態]
    
        style A fill:#fce7f3
        style F fill:#d1fae5
        style H fill:#dbeafe
    ```

## 3\. ギブスの相律（Phase Rule）

### 3.1 相律の導出と意味

ギブスの相律は、平衡状態にある系の**自由度（degrees of freedom）** を決定する重要な関係式です。

#### ギブスの相律

$$ F = C - P + 2 $$ 

  * $F$: **自由度** （独立に変化させられる示強変数の数）
  * $C$: **成分数** （独立な化学成分の数）
  * $P$: **相数** （共存する相の数）
  * $2$: 温度と圧力の2つの示強変数

**自由度 $F$ の意味** : 平衡を保ったまま、独立に変化させられる変数の数。$F = 0$ なら不変系（温度・圧力・組成すべて固定）、$F = 1$ なら一変数系（例: 温度を決めると圧力が決まる）。

### 3.2 相律の適用例

📝 コード例1: 様々な系での相律の検証 コピー
    
    
    import pandas as pd
    
    # ギブスの相律: F = C - P + 2
    
    # 様々な系での自由度計算
    systems = [
        {
            "系": "純水（単相）",
            "成分数 C": 1,
            "相数 P": 1,
            "自由度 F": 1 - 1 + 2,
            "具体例": "液体水のみ → T, P を独立に変えられる"
        },
        {
            "系": "水の沸騰（二相）",
            "成分数 C": 1,
            "相数 P": 2,
            "自由度 F": 1 - 2 + 2,
            "具体例": "液体+気体 → T を決めると P（蒸気圧）が決まる"
        },
        {
            "系": "水の三重点",
            "成分数 C": 1,
            "相数 P": 3,
            "自由度 F": 1 - 3 + 2,
            "具体例": "固体+液体+気体 → T, P とも固定（0.01°C, 611 Pa）"
        },
        {
            "系": "Fe-C合金（単相）",
            "成分数 C": 2,
            "相数 P": 1,
            "自由度 F": 2 - 1 + 2,
            "具体例": "γ-Fe（オーステナイト）のみ → T, P, 組成xを独立に変えられる"
        },
        {
            "系": "Fe-C合金（二相）",
            "成分数 C": 2,
            "相数 P": 2,
            "自由度 F": 2 - 2 + 2,
            "具体例": "α-Fe + Fe₃C → T, P を決めると各相の組成が決まる"
        },
        {
            "系": "Fe-C共晶点",
            "成分数 C": 2,
            "相数 P": 3,
            "自由度 F": 2 - 3 + 2,
            "具体例": "液相 + α-Fe + Fe₃C → T または P を決めると他が決まる"
        }
    ]
    
    df = pd.DataFrame(systems)
    
    print("=" * 80)
    print("ギブスの相律の適用例: F = C - P + 2")
    print("=" * 80)
    print(df.to_string(index=False))
    print("=" * 80)
    
    # 自由度の解釈
    print("\n【自由度の解釈】")
    print("F = 0: 不変系（invariant system）")
    print("       → すべての示強変数が固定（例: 三重点）")
    print("\nF = 1: 一変系（univariant system）")
    print("       → 1つの変数を決めると他が決まる（例: 沸騰曲線）")
    print("\nF = 2: 二変系（bivariant system）")
    print("       → 2つの変数を独立に変えられる（例: 単相領域）")
    print("\nF = 3: 三変系（trivariant system）")
    print("       → 3つの変数を独立に変えられる（例: 二元系の単相）")
    

#### 注意: 相律は平衡状態のみに適用可能

ギブスの相律は、系が**熱力学平衡** にある場合のみ成立します。以下の場合は適用できません：

  * **非平衡状態** : 急冷で得られた準安定相（マルテンサイトなど）
  * **速度論的制約** : 反応が遅く平衡に達していない状態
  * **界面効果** : ナノ粒子など、界面エネルギーが支配的な場合

## 4\. 一成分系の相図

### 4.1 圧力-温度（P-T）相図

一成分系（pure substance）の相図は、**圧力（P）と温度（T）** を軸とした図で表されます。この図には、各相が安定な領域と相境界線が示されます。

📝 コード例2: 水（H₂O）の圧力-温度相図 コピー
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    # 水の相図データ（簡略化モデル）
    # 実際の相図はより複雑ですが、教育目的で簡略化
    
    # 温度範囲 [°C]
    T_solid_liquid = np.array([-100, 0, 0.01])  # 融解曲線
    T_liquid_vapor = np.linspace(0.01, 374, 100)  # 蒸発曲線
    T_solid_vapor = np.linspace(-100, 0.01, 50)  # 昇華曲線
    
    # クラウジウス-クラペイロンの式による蒸気圧（簡略化）
    def vapor_pressure_liquid(T_celsius):
        """液相-気相境界の蒸気圧（近似式）"""
        T = T_celsius + 273.15  # K
        # Antoine式の簡略版
        P = 0.611 * np.exp(17.27 * T_celsius / (T_celsius + 237.3))  # kPa
        return P
    
    def vapor_pressure_solid(T_celsius):
        """固相-気相境界の蒸気圧（昇華圧）"""
        T = T_celsius + 273.15  # K
        # 昇華圧は液相より低い（簡略化）
        P = 0.611 * np.exp(21.87 * T_celsius / (T_celsius + 265.5))  # kPa
        return P
    
    # 圧力計算
    P_solid_liquid = np.array([101.325, 101.325, 0.611])  # 融解曲線（ほぼ垂直）
    P_liquid_vapor = vapor_pressure_liquid(T_liquid_vapor)
    P_solid_vapor = vapor_pressure_solid(T_solid_vapor)
    
    # 可視化
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 相境界線
    ax.plot(T_solid_liquid, P_solid_liquid, 'b-', linewidth=2.5, label='固相-液相境界（融解曲線）')
    ax.plot(T_liquid_vapor, P_liquid_vapor, 'r-', linewidth=2.5, label='液相-気相境界（蒸発曲線）')
    ax.plot(T_solid_vapor, P_solid_vapor, 'g-', linewidth=2.5, label='固相-気相境界（昇華曲線）')
    
    # 三重点
    T_triple = 0.01  # °C
    P_triple = 0.611  # kPa
    ax.plot(T_triple, P_triple, 'ko', markersize=12, label=f'三重点 ({T_triple}°C, {P_triple} kPa)', zorder=10)
    
    # 臨界点
    T_critical = 374  # °C
    P_critical = 22064  # kPa
    ax.plot(T_critical, P_critical, 'rs', markersize=12, label=f'臨界点 ({T_critical}°C, {P_critical/1000:.1f} MPa)', zorder=10)
    
    # 相領域のラベル
    ax.text(-50, 50000, '固相\n（氷）', fontsize=14, ha='center', weight='bold',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    ax.text(100, 50000, '液相\n（水）', fontsize=14, ha='center', weight='bold',
            bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.7))
    ax.text(100, 0.05, '気相\n（水蒸気）', fontsize=14, ha='center', weight='bold',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))
    
    # 大気圧線
    ax.axhline(101.325, color='gray', linestyle='--', linewidth=1.5, alpha=0.7, label='大気圧（101.325 kPa）')
    ax.text(-80, 101.325 * 1.2, '1気圧', fontsize=10, color='gray')
    
    ax.set_xlabel('温度 [°C]', fontsize=13)
    ax.set_ylabel('圧力 [kPa]', fontsize=13)
    ax.set_title('水（H₂O）の圧力-温度相図', fontsize=15, fontweight='bold')
    ax.set_yscale('log')
    ax.set_ylim([0.01, 100000])
    ax.set_xlim([-100, 400])
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    plt.savefig('water_phase_diagram.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("水の相図の重要なポイント：")
    print(f"・三重点: {T_triple}°C, {P_triple} kPa（3相共存、F=0）")
    print(f"・臨界点: {T_critical}°C, {P_critical/1000:.1f} MPa（液相と気相の区別が消失）")
    print(f"・大気圧での沸点: 100°C（液相-気相境界と大気圧線の交点）")
    print(f"・大気圧での融点: 0°C（固相-液相境界と大気圧線の交点）")
    

### 4.2 相図の読み方

#### 相図を読むための基本ルール

  * **領域** : 単一相が安定な領域（固相領域、液相領域、気相領域）
  * **境界線** : 2つの相が共存する線（二相平衡線）、$F = 1$
  * **三重点** : 3つの相が共存する点、$F = 0$（温度・圧力とも固定）
  * **臨界点** : 液相と気相の区別が消失する点（超臨界流体）

#### 実用例: 高圧下での氷の多形

水の相図には、実は**15種類以上の氷の多形** （Ice I, II, III, ..., XV）が存在します。通常の氷（Ice Ih）以外は高圧下でのみ安定です：

  * **Ice Ih** : 大気圧下の通常の氷（六方晶）
  * **Ice III** : 約300 MPa、-20°C付近で安定
  * **Ice VII** : 約2 GPa以上、高密度（地球深部に存在する可能性）

これらは惑星科学や材料物理学で重要な研究対象です。

## 5\. クラペイロンの式とクラウジウス-クラペイロンの式

### 5.1 クラペイロンの式

相境界線の傾き（$dP/dT$）を相転移のエンタルピーとエントロピーで表す重要な関係式です。

#### クラペイロンの式（Clapeyron Equation）

相 α から相 β への転移境界線の傾きは：

$$ \frac{dP}{dT} = \frac{\Delta S_{\text{転移}}}{\Delta V_{\text{転移}}} = \frac{\Delta H_{\text{転移}}}{T \Delta V_{\text{転移}}} $$ 

  * $\Delta H_{\text{転移}}$: 転移エンタルピー（融解熱、蒸発熱など）[J/mol]
  * $\Delta V_{\text{転移}}$: 転移に伴うモル体積変化 [m³/mol]
  * $\Delta S_{\text{転移}}$: 転移エントロピー変化 [J/(mol·K)]
  * $T$: 転移温度 [K]

📝 コード例3: クラペイロンの式による融解曲線の傾き計算 コピー
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    # 水の融解に関するデータ
    T_m = 273.15  # 融点 [K]
    Delta_H_fusion = 6010  # 融解熱 [J/mol]
    V_ice = 19.65e-6  # 氷のモル体積 [m^3/mol]
    V_water = 18.02e-6  # 水のモル体積 [m^3/mol]
    Delta_V = V_water - V_ice  # モル体積変化（負の値！）
    
    # クラペイロンの式による融解曲線の傾き
    dP_dT = Delta_H_fusion / (T_m * Delta_V)  # [Pa/K]
    
    print("=" * 60)
    print("クラペイロンの式による水の融解曲線の傾き計算")
    print("=" * 60)
    print(f"融点（0°C）: {T_m} K")
    print(f"融解熱: {Delta_H_fusion} J/mol = {Delta_H_fusion/1000:.2f} kJ/mol")
    print(f"氷のモル体積: {V_ice * 1e6:.3f} cm³/mol")
    print(f"水のモル体積: {V_water * 1e6:.3f} cm³/mol")
    print(f"モル体積変化 ΔV: {Delta_V * 1e6:.3f} cm³/mol（負の値！）")
    print(f"\n融解曲線の傾き dP/dT: {dP_dT:.2e} Pa/K")
    print(f"                     = {dP_dT / 1e6:.2f} MPa/K")
    print(f"                     = {dP_dT / 101325:.2f} atm/K")
    print("=" * 60)
    
    print("\n【解釈】")
    print("・dP/dT < 0（負の傾き）: 圧力を上げると融点が下がる")
    print("・この異常な挙動は、氷が水より体積が大きいことに起因")
    print("・スケートの滑りやすさは、この圧力融解で説明される（部分的）")
    print("・ほとんどの物質は dP/dT > 0（圧力で融点上昇）")
    
    # 融解曲線の描画
    P_0 = 101325  # 大気圧 [Pa]
    T_range = np.linspace(270, 276, 100)  # [K]
    P_range = P_0 + dP_dT * (T_range - T_m)  # [Pa]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # 左図: 融解曲線
    ax1.plot((T_range - 273.15), P_range / 1e6, linewidth=2.5, color='#3b82f6')
    ax1.axhline(P_0 / 1e6, color='gray', linestyle='--', linewidth=1, alpha=0.7, label='大気圧')
    ax1.axvline(0, color='gray', linestyle='--', linewidth=1, alpha=0.7, label='0°C')
    ax1.plot(0, P_0 / 1e6, 'ro', markersize=10, label='融点（大気圧）', zorder=5)
    ax1.set_xlabel('温度 [°C]', fontsize=12)
    ax1.set_ylabel('圧力 [MPa]', fontsize=12)
    ax1.set_title('水の固相-液相境界（融解曲線）', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([-3, 3])
    
    # 右図: 圧力による融点変化
    pressures = np.linspace(0, 100, 100)  # [MPa]
    T_melt = T_m + pressures * 1e6 / dP_dT  # [K]
    ax2.plot(pressures, T_melt - 273.15, linewidth=2.5, color='#f093fb')
    ax2.axhline(0, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    ax2.set_xlabel('圧力 [MPa]', fontsize=12)
    ax2.set_ylabel('融点 [°C]', fontsize=12)
    ax2.set_title('圧力による融点の変化', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # 注釈: 圧力効果
    ax2.text(50, -0.2, f'100 MPa で融点は\n約{(T_melt[-1] - T_m):.2f} K 低下',
             ha='center', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='#fce7f3', alpha=0.9))
    
    plt.tight_layout()
    plt.savefig('clapeyron_equation_water.png', dpi=150, bbox_inches='tight')
    plt.show()
    

### 5.2 クラウジウス-クラペイロンの式

液相-気相や固相-気相の境界では、気相の体積が液相・固相より圧倒的に大きい（$V_{\text{気}} \gg V_{\text{液・固}}$）ため、クラペイロンの式を簡略化できます。

#### クラウジウス-クラペイロンの式（Clausius-Clapeyron Equation）

液相-気相境界（蒸発曲線）の場合：

$$ \frac{d \ln P}{dT} = \frac{\Delta H_{\text{vap}}}{RT^2} $$ 

または、温度変化 $T_1 \to T_2$ に対する圧力変化：

$$ \ln \frac{P_2}{P_1} = -\frac{\Delta H_{\text{vap}}}{R} \left( \frac{1}{T_2} - \frac{1}{T_1} \right) $$ 

  * $\Delta H_{\text{vap}}$: 蒸発エンタルピー [J/mol]
  * $R = 8.314$ J/(mol·K): 気体定数
  * $P$: 蒸気圧 [Pa]

📝 コード例4: クラウジウス-クラペイロンの式による蒸気圧曲線 コピー
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    # 気体定数
    R = 8.314  # J/(mol·K)
    
    # 様々な物質の蒸発熱データ
    substances = {
        'H₂O': {'Delta_H_vap': 40660, 'T_boil': 373.15, 'P_boil': 101325, 'color': '#3b82f6'},
        'Ethanol': {'Delta_H_vap': 38560, 'T_boil': 351.45, 'P_boil': 101325, 'color': '#10b981'},
        'Acetone': {'Delta_H_vap': 29100, 'T_boil': 329.15, 'P_boil': 101325, 'color': '#f093fb'},
        'Benzene': {'Delta_H_vap': 30720, 'T_boil': 353.25, 'P_boil': 101325, 'color': '#f59e0b'}
    }
    
    # 温度範囲 [K]
    T = np.linspace(250, 400, 200)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # 左図: 蒸気圧の温度依存性（対数スケール）
    for name, data in substances.items():
        Delta_H = data['Delta_H_vap']
        T_boil = data['T_boil']
        P_boil = data['P_boil']
        color = data['color']
    
        # クラウジウス-クラペイロンの式
        ln_P_P0 = -(Delta_H / R) * (1/T - 1/T_boil)
        P = P_boil * np.exp(ln_P_P0)  # [Pa]
    
        ax1.plot(T - 273.15, P / 1000, linewidth=2.5, label=name, color=color)
    
        # 沸点をマーク
        ax1.plot(T_boil - 273.15, P_boil / 1000, 'o', markersize=8, color=color)
    
    ax1.axhline(101.325, color='gray', linestyle='--', linewidth=1.5, alpha=0.7, label='大気圧')
    ax1.set_xlabel('温度 [°C]', fontsize=12)
    ax1.set_ylabel('蒸気圧 [kPa]', fontsize=12)
    ax1.set_title('様々な物質の蒸気圧曲線', fontsize=13, fontweight='bold')
    ax1.set_yscale('log')
    ax1.set_ylim([0.1, 1000])
    ax1.legend(fontsize=10, loc='upper left')
    ax1.grid(True, alpha=0.3, which='both')
    
    # 右図: ln(P) vs 1/T（直線関係の確認）
    for name, data in substances.items():
        Delta_H = data['Delta_H_vap']
        T_boil = data['T_boil']
        P_boil = data['P_boil']
        color = data['color']
    
        T_inv = 1 / T  # [1/K]
        ln_P_P0 = -(Delta_H / R) * (1/T - 1/T_boil)
        ln_P = np.log(P_boil) + ln_P_P0
    
        ax2.plot(T_inv * 1000, ln_P, linewidth=2.5, label=name, color=color)
    
        # 傾きの注釈（最初の物質のみ）
        if name == 'H₂O':
            slope = -Delta_H / R
            ax2.text(3.2, 10, f'傾き = -ΔH_vap/R\n= {slope:.0f} K',
                    fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='#fce7f3', alpha=0.8))
    
    ax2.set_xlabel('1000/T [1000/K]', fontsize=12)
    ax2.set_ylabel('ln(P) [ln(Pa)]', fontsize=12)
    ax2.set_title('クラウジウス-クラペイロンプロット', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10, loc='upper right')
    ax2.grid(True, alpha=0.3)
    ax2.invert_xaxis()  # 温度の増加方向を右にする
    
    plt.tight_layout()
    plt.savefig('clausius_clapeyron_equation.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("=" * 70)
    print("クラウジウス-クラペイロンの式の応用")
    print("=" * 70)
    for name, data in substances.items():
        Delta_H = data['Delta_H_vap']
        T_boil = data['T_boil']
        print(f"\n【{name}】")
        print(f"  沸点（1 atm）: {T_boil - 273.15:.2f} °C")
        print(f"  蒸発熱: {Delta_H/1000:.1f} kJ/mol")
    
        # 80°Cでの蒸気圧を計算
        T_target = 273.15 + 80  # K
        ln_ratio = -(Delta_H / R) * (1/T_target - 1/T_boil)
        P_target = 101325 * np.exp(ln_ratio)
        print(f"  80°C での蒸気圧: {P_target/1000:.2f} kPa")
    
    print("\n" + "=" * 70)
    print("実用例: 減圧蒸留")
    print("蒸気圧が低い（沸点が高い）物質を、減圧することで低温で蒸留できる。")
    print("例: 水を 50 mmHg（約6.7 kPa）で蒸留すると、沸点は約33°Cに下がる。")
    

## 6\. 相転移の分類

### 6.1 エーレンフェストの分類

相転移は、ギブスエネルギーの微分の連続性によって分類されます（エーレンフェスト分類）。

相転移の種類 | 定義 | 特徴 | 具体例  
---|---|---|---  
**一次相転移** | $G$ は連続だが、  
$\frac{\partial G}{\partial T}$ が不連続 | ・潜熱あり  
・体積変化あり  
・二相共存 | 融解、沸騰、昇華、  
同素変態（Fe: α→γ）  
**二次相転移** | $G$, $\frac{\partial G}{\partial T}$ は連続だが、  
$\frac{\partial^2 G}{\partial T^2}$ が不連続 | ・潜熱なし  
・体積変化なし  
・比熱の不連続 | 常磁性-強磁性転移（Fe: 770°C）、  
超伝導転移  
  
#### 一次相転移と二次相転移の見分け方

  * **潜熱の有無** : 一次相転移では、転移温度で熱を吸収・放出（融解熱、蒸発熱など）。二次相転移では潜熱ゼロ。
  * **体積変化** : 一次相転移では相によって密度が異なる（氷→水で体積減少）。二次相転移では連続的。
  * **エントロピー変化** : 一次相転移では $\Delta S = \Delta H / T$ で不連続。二次相転移では連続的。

📝 コード例5: 一次相転移と二次相転移のギブスエネルギー コピー
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    # 温度範囲
    T = np.linspace(200, 400, 500)  # [K]
    T_transition = 300  # 転移温度 [K]
    
    # 一次相転移のモデル（融解を想定）
    Delta_H_fusion = 5000  # J/mol
    Delta_S_fusion = Delta_H_fusion / T_transition  # J/(mol·K)
    
    G_solid_1st = 1000 * (T - 200)  # 固相のギブスエネルギー（簡略化）
    G_liquid_1st = 1000 * (T - 200) - Delta_S_fusion * (T - T_transition)  # 液相
    
    # 安定相の選択（ギブスエネルギーが低い方）
    G_stable_1st = np.minimum(G_solid_1st, G_liquid_1st)
    
    # 二次相転移のモデル（常磁性-強磁性転移を想定）
    # ギブスエネルギーは連続だが、二階微分（比熱）が不連続
    G_paramagnetic = 1000 * (T - 200) + 0.5 * (T - T_transition)**2
    G_ferromagnetic = 1000 * (T - 200) + 2.0 * (T - T_transition)**2
    
    # 安定相の選択
    G_stable_2nd = np.where(T < T_transition, G_ferromagnetic, G_paramagnetic)
    
    # 可視化
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    # ===== 一次相転移 =====
    # 左上: ギブスエネルギー
    axes[0, 0].plot(T, G_solid_1st / 1000, 'b--', linewidth=2, label='固相', alpha=0.7)
    axes[0, 0].plot(T, G_liquid_1st / 1000, 'r--', linewidth=2, label='液相', alpha=0.7)
    axes[0, 0].plot(T, G_stable_1st / 1000, 'k-', linewidth=3, label='安定相')
    axes[0, 0].axvline(T_transition, color='gray', linestyle=':', alpha=0.5)
    axes[0, 0].set_xlabel('温度 [K]', fontsize=11)
    axes[0, 0].set_ylabel('ギブスエネルギー [kJ/mol]', fontsize=11)
    axes[0, 0].set_title('一次相転移: ギブスエネルギー', fontsize=12, fontweight='bold')
    axes[0, 0].legend(fontsize=9)
    axes[0, 0].grid(True, alpha=0.3)
    
    # 中上: エントロピー（一階微分）
    S_solid = -np.gradient(G_solid_1st, T)
    S_liquid = -np.gradient(G_liquid_1st, T)
    S_stable = -np.gradient(G_stable_1st, T)
    axes[0, 1].plot(T, S_solid, 'b--', linewidth=2, alpha=0.7)
    axes[0, 1].plot(T, S_liquid, 'r--', linewidth=2, alpha=0.7)
    axes[0, 1].plot(T, S_stable, 'k-', linewidth=3)
    axes[0, 1].axvline(T_transition, color='gray', linestyle=':', alpha=0.5)
    axes[0, 1].set_xlabel('温度 [K]', fontsize=11)
    axes[0, 1].set_ylabel('エントロピー [J/(mol·K)]', fontsize=11)
    axes[0, 1].set_title('一次相転移: エントロピー（不連続）', fontsize=12, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].text(T_transition + 10, np.mean(S_stable), 'ΔS = ΔH/T\n（潜熱あり）',
                   fontsize=9, bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    # 右上: 比熱（二階微分）
    C_p_solid = T * np.gradient(S_solid, T)
    C_p_liquid = T * np.gradient(S_liquid, T)
    C_p_stable = T * np.gradient(S_stable, T)
    axes[0, 2].plot(T, C_p_stable, 'k-', linewidth=3)
    axes[0, 2].axvline(T_transition, color='gray', linestyle=':', alpha=0.5)
    axes[0, 2].set_xlabel('温度 [K]', fontsize=11)
    axes[0, 2].set_ylabel('比熱 C_p [J/(mol·K)]', fontsize=11)
    axes[0, 2].set_title('一次相転移: 比熱（発散）', fontsize=12, fontweight='bold')
    axes[0, 2].grid(True, alpha=0.3)
    axes[0, 2].set_ylim([-500, 500])
    
    # ===== 二次相転移 =====
    # 左下: ギブスエネルギー
    axes[1, 0].plot(T, G_paramagnetic / 1000, 'b--', linewidth=2, label='常磁性相', alpha=0.7)
    axes[1, 0].plot(T, G_ferromagnetic / 1000, 'r--', linewidth=2, label='強磁性相', alpha=0.7)
    axes[1, 0].plot(T, G_stable_2nd / 1000, 'k-', linewidth=3, label='安定相')
    axes[1, 0].axvline(T_transition, color='gray', linestyle=':', alpha=0.5)
    axes[1, 0].set_xlabel('温度 [K]', fontsize=11)
    axes[1, 0].set_ylabel('ギブスエネルギー [kJ/mol]', fontsize=11)
    axes[1, 0].set_title('二次相転移: ギブスエネルギー（連続）', fontsize=12, fontweight='bold')
    axes[1, 0].legend(fontsize=9)
    axes[1, 0].grid(True, alpha=0.3)
    
    # 中下: エントロピー（一階微分、連続）
    S_para = -np.gradient(G_paramagnetic, T)
    S_ferro = -np.gradient(G_ferromagnetic, T)
    S_stable_2nd = -np.gradient(G_stable_2nd, T)
    axes[1, 1].plot(T, S_stable_2nd, 'k-', linewidth=3)
    axes[1, 1].axvline(T_transition, color='gray', linestyle=':', alpha=0.5)
    axes[1, 1].set_xlabel('温度 [K]', fontsize=11)
    axes[1, 1].set_ylabel('エントロピー [J/(mol·K)]', fontsize=11)
    axes[1, 1].set_title('二次相転移: エントロピー（連続）', fontsize=12, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].text(T_transition + 10, np.mean(S_stable_2nd), 'ΔS = 0\n（潜熱なし）',
                   fontsize=9, bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
    
    # 右下: 比熱（二階微分、不連続）
    C_p_stable_2nd = T * np.gradient(S_stable_2nd, T)
    axes[1, 2].plot(T, C_p_stable_2nd, 'k-', linewidth=3)
    axes[1, 2].axvline(T_transition, color='gray', linestyle=':', alpha=0.5)
    axes[1, 2].set_xlabel('温度 [K]', fontsize=11)
    axes[1, 2].set_ylabel('比熱 C_p [J/(mol·K)]', fontsize=11)
    axes[1, 2].set_title('二次相転移: 比熱（不連続）', fontsize=12, fontweight='bold')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('phase_transition_classification.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("相転移の分類まとめ：")
    print("\n【一次相転移】")
    print("・G: 連続、S = -∂G/∂T: 不連続（潜熱あり）")
    print("・例: 融解（固相→液相）、蒸発（液相→気相）")
    print("・測定: DSC（示差走査熱量計）でピークが現れる")
    print("\n【二次相転移】")
    print("・G: 連続、S: 連続、C_p = T ∂S/∂T: 不連続")
    print("・例: 強磁性-常磁性転移（Feのキュリー温度: 770°C）")
    print("・測定: DSCで比熱の段差が現れる（ピークではない）")
    

## 7\. レバールール（Lever Rule）

### 7.1 レバールールの原理

二相領域において、系の平均組成が与えられたとき、各相の量比（相分率）は**レバールール（lever rule）** で決まります。これは「てこの原理」と数学的に同じ関係です。

#### レバールールの公式

二元系で、α相の組成 $x_\alpha$、β相の組成 $x_\beta$、系の平均組成 $x_{\text{avg}}$ が $x_\alpha < x_{\text{avg}} < x_\beta$ のとき：

$$ \frac{n_\beta}{n_\alpha} = \frac{x_{\text{avg}} - x_\alpha}{x_\beta - x_{\text{avg}}} $$ 

または、各相のモル分率（質量分率）で：

$$ f_\alpha = \frac{x_\beta - x_{\text{avg}}}{x_\beta - x_\alpha}, \quad f_\beta = \frac{x_{\text{avg}} - x_\alpha}{x_\beta - x_\alpha} $$ 

ここで、$f_\alpha + f_\beta = 1$ です。

#### レバールールの幾何学的意味

てこ（lever）を想像してください。支点（fulcrum）が平均組成 $x_{\text{avg}}$、左端が $x_\alpha$、右端が $x_\beta$ です。

  * **左のアーム（α相側）の長さ** : $x_{\text{avg}} - x_\alpha$
  * **右のアーム（β相側）の長さ** : $x_\beta - x_{\text{avg}}$

てこの釣り合い条件から：$n_\alpha \times (x_{\text{avg}} - x_\alpha) = n_\beta \times (x_\beta - x_{\text{avg}})$

これを整理すると、上記のレバールールの公式になります。

📝 コード例6: レバールールの計算と可視化 コピー
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    # 二相領域の設定
    x_alpha = 0.2  # α相の組成
    x_beta = 0.8   # β相の組成
    
    # 平均組成の範囲（二相領域内）
    x_avg_range = np.linspace(x_alpha, x_beta, 100)
    
    # レバールールによる相分率計算
    def lever_rule(x_avg, x_alpha, x_beta):
        """レバールールで各相の分率を計算"""
        f_alpha = (x_beta - x_avg) / (x_beta - x_alpha)
        f_beta = (x_avg - x_alpha) / (x_beta - x_alpha)
        return f_alpha, f_beta
    
    f_alpha_range = []
    f_beta_range = []
    for x_avg in x_avg_range:
        f_a, f_b = lever_rule(x_avg, x_alpha, x_beta)
        f_alpha_range.append(f_a)
        f_beta_range.append(f_b)
    
    f_alpha_range = np.array(f_alpha_range)
    f_beta_range = np.array(f_beta_range)
    
    # 可視化
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # 左図: 相分率の変化
    ax1.plot(x_avg_range, f_alpha_range, linewidth=3, color='#3b82f6', label='α相の分率 $f_\\alpha$')
    ax1.plot(x_avg_range, f_beta_range, linewidth=3, color='#f5576c', label='β相の分率 $f_\\beta$')
    ax1.fill_between(x_avg_range, 0, f_alpha_range, alpha=0.3, color='#3b82f6')
    ax1.fill_between(x_avg_range, f_alpha_range, 1, alpha=0.3, color='#f5576c')
    
    # 境界線
    ax1.axvline(x_alpha, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
    ax1.axvline(x_beta, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
    ax1.text(x_alpha, 1.05, f'$x_\\alpha={x_alpha}$', ha='center', fontsize=10)
    ax1.text(x_beta, 1.05, f'$x_\\beta={x_beta}$', ha='center', fontsize=10)
    
    # 具体例をマーク
    x_example = 0.5
    f_a_ex, f_b_ex = lever_rule(x_example, x_alpha, x_beta)
    ax1.plot(x_example, f_a_ex, 'o', markersize=12, color='#10b981', zorder=5)
    ax1.plot(x_example, f_b_ex, 'o', markersize=12, color='#10b981', zorder=5)
    ax1.text(x_example + 0.05, f_a_ex, f'$x_{{avg}}={x_example}$\n$f_\\alpha={f_a_ex:.2f}$\n$f_\\beta={f_b_ex:.2f}$',
             fontsize=9, bbox=dict(boxstyle='round', facecolor='#d1fae5', alpha=0.9))
    
    ax1.set_xlabel('平均組成 $x_{avg}$', fontsize=12)
    ax1.set_ylabel('相分率 [-]', fontsize=12)
    ax1.set_title('レバールールによる相分率の計算', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=11, loc='center left')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([x_alpha - 0.05, x_beta + 0.05])
    ax1.set_ylim([0, 1.1])
    
    # 右図: てこの原理の図解
    ax2.axis('off')
    ax2.set_xlim([0, 1])
    ax2.set_ylim([0, 1])
    
    # てこの描画
    lever_y = 0.6
    ax2.plot([x_alpha, x_beta], [lever_y, lever_y], 'k-', linewidth=4)
    ax2.plot([x_alpha], [lever_y], 'o', markersize=15, color='#3b82f6', label='α相')
    ax2.plot([x_beta], [lever_y], 'o', markersize=15, color='#f5576c', label='β相')
    ax2.plot([x_example], [lever_y], '^', markersize=15, color='#10b981', label='支点（平均組成）')
    
    # 距離の表示
    ax2.plot([x_alpha, x_example], [lever_y - 0.1, lever_y - 0.1], 'b-', linewidth=2)
    ax2.text((x_alpha + x_example) / 2, lever_y - 0.15, f'${x_example - x_alpha:.1f}$',
             ha='center', fontsize=11, color='blue')
    ax2.plot([x_example, x_beta], [lever_y - 0.1, lever_y - 0.1], 'r-', linewidth=2)
    ax2.text((x_example + x_beta) / 2, lever_y - 0.15, f'${x_beta - x_example:.1f}$',
             ha='center', fontsize=11, color='red')
    
    # 力の矢印（相の量に対応）
    arrow_y = 0.75
    ax2.arrow(x_alpha, arrow_y, 0, -0.08, head_width=0.03, head_length=0.03,
              fc='#3b82f6', ec='#3b82f6', linewidth=2)
    ax2.text(x_alpha, arrow_y + 0.05, f'$n_\\alpha$ または $f_\\alpha={f_a_ex:.2f}$',
             ha='center', fontsize=10, color='#3b82f6')
    
    ax2.arrow(x_beta, arrow_y, 0, -0.08, head_width=0.03, head_length=0.03,
              fc='#f5576c', ec='#f5576c', linewidth=2)
    ax2.text(x_beta, arrow_y + 0.05, f'$n_\\beta$ または $f_\\beta={f_b_ex:.2f}$',
             ha='center', fontsize=10, color='#f5576c')
    
    # 説明文
    ax2.text(0.5, 0.35, 'てこの釣り合い条件:\n$n_\\alpha \\times$ (右のアーム) $= n_\\beta \\times$ (左のアーム)',
             ha='center', fontsize=11,
             bbox=dict(boxstyle='round', facecolor='#fef3c7', alpha=0.9))
    ax2.text(0.5, 0.2, f'$\\frac{{n_\\beta}}{{n_\\alpha}} = \\frac{{{x_example - x_alpha:.1f}}}{{{x_beta - x_example:.1f}}} = {f_b_ex / f_a_ex:.2f}$',
             ha='center', fontsize=12)
    
    ax2.set_title('レバールールの幾何学的意味', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10, loc='upper left')
    
    plt.tight_layout()
    plt.savefig('lever_rule.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # 具体的な計算例
    print("=" * 70)
    print("レバールール計算例")
    print("=" * 70)
    print(f"α相の組成: x_α = {x_alpha}")
    print(f"β相の組成: x_β = {x_beta}")
    print(f"\n【ケース1】平均組成 x_avg = {x_example}")
    print(f"  α相の分率: f_α = (x_β - x_avg) / (x_β - x_α) = {f_a_ex:.3f} ({f_a_ex*100:.1f}%)")
    print(f"  β相の分率: f_β = (x_avg - x_α) / (x_β - x_α) = {f_b_ex:.3f} ({f_b_ex*100:.1f}%)")
    print(f"  検証: f_α + f_β = {f_a_ex + f_b_ex:.3f} ✓")
    
    x_example2 = 0.3
    f_a_ex2, f_b_ex2 = lever_rule(x_example2, x_alpha, x_beta)
    print(f"\n【ケース2】平均組成 x_avg = {x_example2} (α相寄り)")
    print(f"  α相の分率: f_α = {f_a_ex2:.3f} ({f_a_ex2*100:.1f}%)")
    print(f"  β相の分率: f_β = {f_b_ex2:.3f} ({f_b_ex2*100:.1f}%)")
    print(f"  → α相が多い（平均組成がα相に近いため）")
    
    x_example3 = 0.7
    f_a_ex3, f_b_ex3 = lever_rule(x_example3, x_alpha, x_beta)
    print(f"\n【ケース3】平均組成 x_avg = {x_example3} (β相寄り)")
    print(f"  α相の分率: f_α = {f_a_ex3:.3f} ({f_a_ex3*100:.1f}%)")
    print(f"  β相の分率: f_β = {f_b_ex3:.3f} ({f_b_ex3*100:.1f}%)")
    print(f"  → β相が多い（平均組成がβ相に近いため）")
    print("=" * 70)
    

### 7.2 レバールールの実用例

#### 鋼の炭素量と組織

Fe-C合金（鋼）で、727°C（共析温度）での組織を考えます：

  * **α-Fe（フェライト）** : C濃度 0.02 wt%
  * **Fe₃C（セメンタイト）** : C濃度 6.7 wt%
  * **共析鋼** : C濃度 0.76 wt%

レバールールで、共析鋼中のフェライトとセメンタイトの質量比を計算：

$f_{\text{Fe}_3\text{C}} = \frac{0.76 - 0.02}{6.7 - 0.02} = \frac{0.74}{6.68} \approx 0.11$ (11%)

$f_{\alpha\text{-Fe}} = 1 - 0.11 = 0.89$ (89%)

つまり、共析鋼は約89%のフェライトと11%のセメンタイトで構成されます。

## 8\. 相変態エンタルピーの計算

📝 コード例7: 三重点の決定と可視化 コピー
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.optimize import fsolve
    
    # 水の相転移データ
    data = {
        '融解': {
            'Delta_H': 6010,  # J/mol
            'T_0': 273.15,    # K (0°C)
            'P_0': 101325,    # Pa
            'Delta_V': (18.02e-6 - 19.65e-6),  # m^3/mol（液-固）
        },
        '蒸発': {
            'Delta_H': 40660,  # J/mol
            'T_0': 373.15,     # K (100°C)
            'P_0': 101325,     # Pa
        },
        '昇華': {
            'Delta_H': 51000,  # J/mol (融解+蒸発の和に近い)
            'T_0': 273.15,     # K
            'P_0': 611,        # Pa（推定）
        }
    }
    
    R = 8.314  # J/(mol·K)
    
    # 各境界線の方程式
    def melting_curve(T):
        """固-液境界（クラペイロン）"""
        dP_dT = data['融解']['Delta_H'] / (data['融解']['T_0'] * data['融解']['Delta_V'])
        P = data['融解']['P_0'] + dP_dT * (T - data['融解']['T_0'])
        return P
    
    def vaporization_curve(T):
        """液-気境界（クラウジウス-クラペイロン）"""
        Delta_H = data['蒸発']['Delta_H']
        T_0 = data['蒸発']['T_0']
        P_0 = data['蒸発']['P_0']
        ln_ratio = -(Delta_H / R) * (1/T - 1/T_0)
        P = P_0 * np.exp(ln_ratio)
        return P
    
    def sublimation_curve(T):
        """固-気境界（クラウジウス-クラペイロン）"""
        Delta_H = data['昇華']['Delta_H']
        T_0 = data['昇華']['T_0']
        P_0 = data['昇華']['P_0']
        ln_ratio = -(Delta_H / R) * (1/T - 1/T_0)
        P = P_0 * np.exp(ln_ratio)
        return P
    
    # 三重点の決定（融解曲線と昇華曲線の交点）
    def triple_point_equation(T):
        """三重点での条件: 融解曲線の圧力 = 昇華曲線の圧力"""
        return melting_curve(T) - sublimation_curve(T)
    
    T_triple = fsolve(triple_point_equation, 273.15)[0]
    P_triple = sublimation_curve(T_triple)
    
    print("=" * 60)
    print("三重点の計算結果")
    print("=" * 60)
    print(f"温度: {T_triple:.2f} K = {T_triple - 273.15:.2f} °C")
    print(f"圧力: {P_triple:.2f} Pa = {P_triple / 1000:.3f} kPa")
    print("\n実験値（文献値）:")
    print("温度: 273.16 K = 0.01 °C")
    print("圧力: 611.657 Pa = 0.612 kPa")
    print("\n→ 計算値は実験値とよく一致")
    print("=" * 60)
    
    # 温度範囲
    T_range = np.linspace(250, 400, 300)
    P_melt = melting_curve(T_range)
    P_vap = vaporization_curve(T_range)
    P_sub = sublimation_curve(T_range)
    
    # 可視化
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 相境界線
    T_melt_range = np.linspace(260, 280, 100)
    T_vap_range = np.linspace(273, 374, 100)
    T_sub_range = np.linspace(250, 273.16, 100)
    
    ax.plot(T_melt_range - 273.15, melting_curve(T_melt_range) / 1000,
            'b-', linewidth=3, label='固-液境界')
    ax.plot(T_vap_range - 273.15, vaporization_curve(T_vap_range) / 1000,
            'r-', linewidth=3, label='液-気境界')
    ax.plot(T_sub_range - 273.15, sublimation_curve(T_sub_range) / 1000,
            'g-', linewidth=3, label='固-気境界')
    
    # 三重点
    ax.plot(T_triple - 273.15, P_triple / 1000, 'ko', markersize=14, zorder=10,
            label=f'三重点\n({T_triple - 273.15:.2f}°C, {P_triple / 1000:.3f} kPa)')
    
    # 相律の表示
    ax.text(-15, 50, 'F = 2\n(単相領域)', fontsize=11, ha='center',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    ax.text(50, 50, 'F = 1\n(二相共存線)', fontsize=11, ha='center',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    ax.text(T_triple - 273.15 - 5, P_triple / 1000 + 0.3, 'F = 0\n(三相共存点)',
            fontsize=10, ha='right',
            bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
    
    ax.set_xlabel('温度 [°C]', fontsize=13)
    ax.set_ylabel('圧力 [kPa]', fontsize=13)
    ax.set_title('水の相図と三重点（計算結果）', fontsize=15, fontweight='bold')
    ax.set_yscale('log')
    ax.set_ylim([0.1, 200])
    ax.set_xlim([-25, 105])
    ax.legend(fontsize=11, loc='upper left')
    ax.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    plt.savefig('triple_point_calculation.png', dpi=150, bbox_inches='tight')
    plt.show()
    

## 9\. 実材料の相転移温度計算

📝 コード例8: Fe, Tiの同素変態温度計算 コピー
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    # 純金属の同素変態データ
    metals = {
        'Fe (α→γ)': {
            'T_trans': 912 + 273.15,  # K
            'Delta_H': 900,           # J/mol
            'phase_low': 'α-Fe (BCC)',
            'phase_high': 'γ-Fe (FCC)',
            'color': '#f5576c'
        },
        'Fe (γ→δ)': {
            'T_trans': 1394 + 273.15,  # K
            'Delta_H': 840,            # J/mol
            'phase_low': 'γ-Fe (FCC)',
            'phase_high': 'δ-Fe (BCC)',
            'color': '#f093fb'
        },
        'Ti (α→β)': {
            'T_trans': 882 + 273.15,   # K
            'Delta_H': 4000,           # J/mol
            'phase_low': 'α-Ti (HCP)',
            'phase_high': 'β-Ti (BCC)',
            'color': '#3b82f6'
        },
        'Co (ε→α)': {
            'T_trans': 422 + 273.15,   # K
            'Delta_H': 450,            # J/mol
            'phase_low': 'ε-Co (HCP)',
            'phase_high': 'α-Co (FCC)',
            'color': '#10b981'
        }
    }
    
    R = 8.314  # J/(mol·K)
    
    # 相転移温度の圧力依存性計算（クラペイロン近似）
    # 簡略化: ΔV を推定（典型値として 0.1 cm³/mol = 1e-7 m³/mol）
    Delta_V_typical = 1e-7  # m³/mol
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 左図: 同素変態温度の比較
    metal_names = []
    T_trans_list = []
    Delta_H_list = []
    colors = []
    
    for name, data in metals.items():
        metal_names.append(name)
        T_trans_list.append(data['T_trans'] - 273.15)  # °C
        Delta_H_list.append(data['Delta_H'] / 1000)    # kJ/mol
        colors.append(data['color'])
    
    x_pos = np.arange(len(metal_names))
    ax1.bar(x_pos, T_trans_list, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(metal_names, rotation=15, ha='right', fontsize=10)
    ax1.set_ylabel('変態温度 [°C]', fontsize=12)
    ax1.set_title('純金属の同素変態温度', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 各バーに数値を表示
    for i, (T, DH) in enumerate(zip(T_trans_list, Delta_H_list)):
        ax1.text(i, T + 30, f'{T:.0f}°C\nΔH={DH:.1f} kJ/mol',
                 ha='center', fontsize=9,
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 右図: 変態エンタルピーと温度の関係
    ax2.scatter(T_trans_list, Delta_H_list, s=200, c=colors, alpha=0.7,
                edgecolor='black', linewidth=2)
    for name, T, DH, color in zip(metal_names, T_trans_list, Delta_H_list, colors):
        ax2.annotate(name, (T, DH), xytext=(10, 10), textcoords='offset points',
                    fontsize=9,
                    bbox=dict(boxstyle='round', facecolor=color, alpha=0.3),
                    arrowprops=dict(arrowstyle='->', color='gray'))
    
    ax2.set_xlabel('変態温度 [°C]', fontsize=12)
    ax2.set_ylabel('変態エンタルピー ΔH [kJ/mol]', fontsize=12)
    ax2.set_title('変態温度とエンタルピーの関係', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('metal_allotropic_transformation.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # 詳細データの表示
    print("=" * 80)
    print("純金属の同素変態データ")
    print("=" * 80)
    for name, data in metals.items():
        print(f"\n【{name}】")
        print(f"  変態温度: {data['T_trans'] - 273.15:.0f}°C ({data['T_trans']:.2f} K)")
        print(f"  変態エンタルピー: {data['Delta_H']/1000:.2f} kJ/mol")
        print(f"  低温相: {data['phase_low']}")
        print(f"  高温相: {data['phase_high']}")
    
        # 変態エントロピーの計算
        Delta_S = data['Delta_H'] / data['T_trans']  # J/(mol·K)
        print(f"  変態エントロピー: ΔS = ΔH/T = {Delta_S:.2f} J/(mol·K)")
    
    print("\n" + "=" * 80)
    print("同素変態の意義")
    print("=" * 80)
    print("・鉄の α→γ 変態: 焼入れ・焼戻しによる鋼の熱処理の基礎")
    print("・チタンの α→β 変態: 高温での塑性加工性向上（β加工）")
    print("・コバルトの ε→α 変態: 磁性材料としての応用")
    print("・変態温度は合金元素添加で制御可能（状態図工学の基礎）")
    
    # 圧力効果の簡単な推定
    print("\n" + "=" * 80)
    print("圧力による変態温度の変化（推定）")
    print("=" * 80)
    for name, data in metals.items():
        T_trans = data['T_trans']
        Delta_H = data['Delta_H']
        # クラペイロン: dT/dP = T ΔV / ΔH
        dT_dP = T_trans * Delta_V_typical / Delta_H  # K/Pa
        dT_dP_MPa = dT_dP * 1e6  # K/MPa
    
        print(f"{name}: dT/dP ≈ {dT_dP_MPa:.3f} K/MPa")
        print(f"  → 100 MPa で約 {dT_dP_MPa * 100:.1f} K の変化")
    

#### 注意: 実際の相転移温度の圧力依存性

上記の計算は教育目的の簡略化です。実際には：

  * **ΔVの精密測定** が必要（X線回折などで決定）
  * **圧力依存性** : ΔHやΔVも圧力で変化する
  * **高圧相** : 数GPa以上で新しい相が現れる（例: Fe-ε相）

精密な計算には、CALPHADデータベースや第一原理計算が用いられます。

## まとめ

#### 本章で学んだ重要事項

  * **相の定義** : 組成・物性が均一で、明確な界面で区切られた領域
  * **平衡条件** : 温度・圧力・化学ポテンシャルが全相で等しい
  * **ギブスの相律** : $F = C - P + 2$、系の自由度を決定
  * **一成分系相図** : P-T図で固相・液相・気相の安定領域を表示
  * **クラペイロンの式** : 相境界線の傾き $dP/dT = \Delta H / (T \Delta V)$
  * **クラウジウス-クラペイロンの式** : 液-気、固-気境界の簡略式
  * **相転移の分類** : 一次相転移（潜熱あり）、二次相転移（潜熱なし）
  * **レバールール** : 二相領域での各相の量比を計算
  * **実材料への応用** : 水、Fe、Tiなどの相転移温度と圧力依存性

次章では、二元系相図（binary phase diagram）に進み、より複雑な合金系での相平衡を学びます。共晶反応、包晶反応、固溶体の相図読解など、実用材料の設計に直結する内容を扱います。

### 📝 演習問題

#### 演習1: ギブスの相律の適用

**問題** : 以下の系の自由度 $F$ を計算してください。

  * (a) 純アルミニウムの液相のみが存在する状態
  * (b) 純アルミニウムの固相と液相が共存する状態（融解中）
  * (c) Cu-Zn合金（黄銅）の単相（α相）領域
  * (d) Cu-Zn合金の二相（α相+β相）領域

**ヒント** : $F = C - P + 2$ を使います。一定圧力下（大気圧）なら $F = C - P + 1$ です。

#### 演習2: クラペイロンの式の応用

**問題** : 純鉄の α→γ 変態について、以下のデータを用いて変態温度の圧力依存性 $dT/dP$ を計算してください。

  * 変態温度: $T_{\text{trans}} = 912°\text{C} = 1185$ K
  * 変態エンタルピー: $\Delta H = 900$ J/mol
  * モル体積変化: $\Delta V = V_\gamma - V_\alpha = 0.05 \times 10^{-6}$ m³/mol

**ヒント** : $dP/dT = \Delta H / (T \Delta V)$ を使い、逆数を取って $dT/dP$ を求めます。

#### 演習3: レバールールの計算

**問題** : Fe-C合金（炭素鋼）で、727°Cでの共析反応を考えます。以下の条件で、α-Fe（フェライト）とFe₃C（セメンタイト）の質量比を計算してください。

  * α-Feの炭素濃度: 0.02 wt% C
  * Fe₃Cの炭素濃度: 6.7 wt% C
  * 合金全体の炭素濃度: 0.4 wt% C（亜共析鋼）

**ヒント** : レバールール $f_{\text{Fe}_3\text{C}} = (x_{\text{avg}} - x_\alpha) / (x_{\text{Fe}_3\text{C}} - x_\alpha)$ を使います。

#### 演習4: 蒸気圧曲線の計算（発展）

**問題** : エタノールの蒸気圧を、以下のデータを用いて20°Cと60°Cで計算してください。

  * 沸点（1気圧）: $T_{\text{boil}} = 78.3°\text{C}$
  * 蒸発エンタルピー: $\Delta H_{\text{vap}} = 38560$ J/mol

**ヒント** : クラウジウス-クラペイロンの式 $\ln(P_2 / P_1) = -(\Delta H_{\text{vap}} / R)(1/T_2 - 1/T_1)$ を使います。$P_1 = 101325$ Pa（1気圧）、$T_1 = 78.3 + 273.15$ K です。

## 参考文献

  1. D.R. Gaskell, D.E. Laughlin, "Introduction to the Thermodynamics of Materials", 6th Edition, CRC Press, 2017
  2. D.A. Porter, K.E. Easterling, M.Y. Sherif, "Phase Transformations in Metals and Alloys", 3rd Edition, CRC Press, 2009
  3. P. Atkins, J. de Paula, "Atkins' Physical Chemistry", 11th Edition, Oxford University Press, 2018
  4. H.L. Lukas, S.G. Fries, B. Sundman, "Computational Thermodynamics: The CALPHAD Method", Cambridge University Press, 2007
  5. J.W. Christian, "The Theory of Transformations in Metals and Alloys", 3rd Edition, Pergamon Press, 2002
  6. 幸田成康, 「改訂 金属物理学序論」, コロナ社, 1973
  7. 西澤泰二, 「材料組織学」, 朝倉書店, 2005

[← 前の章：ギブスエネルギーと化学ポテンシャル](<chapter-2.html>) 次の章へ：二元系相図と相平衡 →（準備中）
