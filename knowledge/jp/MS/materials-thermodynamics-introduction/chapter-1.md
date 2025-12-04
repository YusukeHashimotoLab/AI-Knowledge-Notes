---
title: 第1章：熱力学の第0法則・第1法則
chapter_title: 第1章：熱力学の第0法則・第1法則
subtitle: 温度平衡とエネルギー保存から材料物性を理解する
reading_time: 30-35分
difficulty: 中級
code_examples: 4
---

熱力学の第0法則と第1法則は、材料科学の基礎を成す重要な原理です。温度とは何か、エネルギーはどのように保存されるのか、そして材料の熱的性質をどのように理解し測定するのかを学びます。 

## 学習目標

この章を読むことで、以下を習得できます：

### 基本理解（Basic Understanding）

  * ✅ 熱力学の第0法則の意味と温度の定義を説明できる
  * ✅ 熱力学の第1法則（エネルギー保存則）を理解し、熱と仕事の関係を説明できる
  * ✅ 内部エネルギー、熱容量、エンタルピーの概念を理解する
  * ✅ 材料の相転移における熱力学的変化を説明できる

### 実践スキル（Practical Skills）

  * ✅ 熱容量と比熱の測定原理を理解し、計算できる
  * ✅ 相転移における潜熱を計算できる
  * ✅ Pythonで材料の熱的性質をモデル化し可視化できる
  * ✅ 熱量測定（カロリメトリ）のデータを解析できる

### 応用力（Application）

  * ✅ 実際の材料（金属、セラミックス、高分子）の熱的挙動を予測できる
  * ✅ 熱処理プロセスにおけるエネルギー収支を計算できる
  * ✅ 材料選択において熱的性質を考慮した判断ができる

* * *

## 1.1 熱力学の第0法則：温度と熱平衡

### 温度とは何か

私たちは日常的に「温度」という概念を使っていますが、厳密に定義すると、**熱力学的平衡状態を特徴づける状態量** です。熱力学の第0法則は、この温度概念の基礎を与えます。

#### 熱力学の第0法則（Zeroth Law of Thermodynamics）

> 物体Aと物体Bが熱平衡にあり、物体Bと物体Cが熱平衡にあるならば、物体Aと物体Cも熱平衡にある。 

**意味** : この法則により、「温度」という状態量を定義でき、温度計を用いた測定が可能になります。

### 熱平衡と温度測定

**熱平衡（Thermal Equilibrium）** とは、2つの系が熱的に接触しているときに、正味の熱エネルギー移動がない状態です。この状態では、両方の系は同じ温度を持ちます。

**温度測定の原理** :

  1. **温度計（物体B）** を測定対象（物体A）と接触させる
  2. 十分な時間待ち、熱平衡に到達させる
  3. 温度計の示す値（例：水銀の体積変化）から温度を読み取る

### 温度スケール

温度スケール | 記号 | 水の凝固点 | 水の沸点 | 絶対零度 | 用途  
---|---|---|---|---|---  
**ケルビン（絶対温度）** | K | 273.15 K | 373.15 K | 0 K | 科学・工学の標準  
**摂氏** | °C | 0 °C | 100 °C | −273.15 °C | 日常、実用  
**華氏** | °F | 32 °F | 212 °F | −459.67 °F | 米国、一部地域  
  
**変換式** :

$$T[\text{K}] = T[^\circ\text{C}] + 273.15$$

$$T[^\circ\text{F}] = \frac{9}{5}T[^\circ\text{C}] + 32$$

#### 💡 材料科学における温度の重要性

材料の多くの性質（強度、電気伝導率、拡散係数など）は温度に強く依存します。例えば：

  * **鋼の焼入れ** : 900-1000°Cに加熱後、急冷して硬度を上げる
  * **半導体プロセス** : 精密な温度制御（±1°C以下）が必須
  * **高分子材料** : ガラス転移温度（Tg）が使用温度範囲を決定

### 材料の熱膨張

温度変化により材料の寸法が変化する現象を**熱膨張（Thermal Expansion）** といいます。

**線膨張係数（Linear Thermal Expansion Coefficient）** :

$$\alpha = \frac{1}{L_0}\frac{dL}{dT}$$

ここで、$L_0$ は基準長さ、$dL$ は長さ変化、$dT$ は温度変化です。

材料 | 線膨張係数 α (× 10⁻⁶ K⁻¹) | 用途への影響  
---|---|---  
インバー合金（Fe-36%Ni） | 1.2 | 精密測定器、時計部品（低膨張）  
ガラス（石英） | 0.5 | 光学機器、実験器具  
鉄鋼 | 11-13 | 建築、機械構造  
アルミニウム | 23 | 軽量構造（熱膨張に注意）  
ポリエチレン | 100-200 | 包装材（大きな熱膨張）  
  
* * *

## 1.2 熱力学の第1法則：エネルギー保存則

### 第1法則の定義

#### 熱力学の第1法則（First Law of Thermodynamics）

系の内部エネルギー変化 $\Delta U$ は、系に加えられた熱 $Q$ と系がされた仕事 $W$ の和に等しい：

$$\Delta U = Q - W$$

または微分形式で：

$$dU = \delta Q - \delta W$$

ここで、$\delta$ は経路依存（不完全微分）を表します。

**符号規約** :

  * $Q > 0$: 系が熱を**吸収** （吸熱）
  * $Q < 0$: 系が熱を**放出** （発熱）
  * $W > 0$: 系が外部に**仕事をする** （膨張）
  * $W < 0$: 系が外部から**仕事をされる** （圧縮）

### 内部エネルギー（Internal Energy）

**内部エネルギー $U$** は、系を構成する粒子（原子、分子）の運動エネルギーとポテンシャルエネルギーの総和です：

$$U = U_{\text{kinetic}} + U_{\text{potential}}$$

**重要な性質** :

  * 内部エネルギーは**状態量** （経路に依存しない）
  * 絶対値は不明だが、変化量 $\Delta U$ は測定可能
  * 理想気体では温度のみの関数: $U = U(T)$

### 熱と仕事

**熱（Heat）$Q$** : 温度差により系と外界の間で移動するエネルギー

**仕事（Work）$W$** : 力学的な相互作用により移動するエネルギー

**圧力−体積仕事（PV Work）** :

気体が膨張・圧縮する際の仕事：

$$W = \int_{V_1}^{V_2} P \, dV$$

#### 例題1.1：理想気体の等温膨張

**問題** : 1 mol の理想気体が、温度 300 K で、体積を 10 L から 20 L へ等温膨張するとき、系がする仕事 $W$ と吸収する熱 $Q$ を求めよ。

解答を見る

**解答** :

等温過程では $PV = nRT = \text{const}$ より：

$$W = \int_{V_1}^{V_2} P \, dV = nRT \int_{V_1}^{V_2} \frac{dV}{V} = nRT \ln\frac{V_2}{V_1}$$

数値代入：

$$W = (1 \text{ mol})(8.314 \text{ J/(mol·K)})(300 \text{ K}) \ln\frac{20}{10}$$

$$W = 2494 \times 0.693 = 1729 \text{ J}$$

等温過程では $\Delta U = 0$（理想気体の内部エネルギーは温度のみの関数）

第1法則より：$Q = \Delta U + W = 0 + 1729 = 1729 \text{ J}$

**答** : 系は 1729 J の仕事をし、同量の熱を吸収する。

* * *

## 1.3 熱容量と比熱

### 熱容量（Heat Capacity）

**熱容量 $C$** は、系の温度を 1 K 上昇させるのに必要な熱量です：

$$C = \frac{\delta Q}{dT}$$

単位は J/K または J/(mol·K)

### 比熱（Specific Heat Capacity）

**比熱 $c$** は、単位質量あたりの熱容量です：

$$c = \frac{C}{m}$$

単位は J/(kg·K) または J/(g·K)

### 定圧比熱と定容比熱

熱容量は、測定条件（圧力一定 or 体積一定）により異なります：

**定容熱容量（Constant Volume Heat Capacity）** :

$$C_V = \left(\frac{\partial U}{\partial T}\right)_V$$

**定圧熱容量（Constant Pressure Heat Capacity）** :

$$C_P = \left(\frac{\partial H}{\partial T}\right)_P$$

ここで $H = U + PV$ は**エンタルピー（Enthalpy）** です。

**理想気体の場合** :

$$C_P - C_V = nR$$

$$\gamma = \frac{C_P}{C_V}$$

$\gamma$ は**比熱比（Heat Capacity Ratio）** または**断熱指数（Adiabatic Index）** と呼ばれます。

気体の種類 | 自由度 | $C_V$ | $C_P$ | $\gamma$  
---|---|---|---|---  
単原子分子（He, Ar） | 3（並進） | $\frac{3}{2}R$ | $\frac{5}{2}R$ | 1.67  
二原子分子（N₂, O₂） | 5（並進3+回転2） | $\frac{5}{2}R$ | $\frac{7}{2}R$ | 1.40  
多原子分子（CO₂, CH₄） | 6以上 | $\geq 3R$ | $\geq 4R$ | 1.29-1.33  
  
### 固体の熱容量

**古典論（Dulong-Petit の法則）** :

高温での原子あたりのモル熱容量：

$$C_V = 3R \approx 25 \text{ J/(mol·K)}$$

**Debye モデル** :

低温での熱容量の温度依存性：

$$C_V \propto T^3 \quad (T \ll \Theta_D)$$

ここで $\Theta_D$ は**Debye 温度** です。

材料 | 比熱 c (J/(g·K)) | Debye温度 Θ_D (K) | 特徴  
---|---|---|---  
ダイヤモンド（C） | 0.51 | 2230 | 極めて硬い結合  
銅（Cu） | 0.385 | 343 | 高い熱伝導率  
鉄（Fe） | 0.449 | 470 | 構造材料  
アルミニウム（Al） | 0.900 | 428 | 軽量、高比熱  
水（H₂O） | 4.18 | − | 最も高い比熱  
  
* * *

## 1.4 相転移と潜熱

### 相転移（Phase Transition）

物質が固体、液体、気体の間で状態を変える現象を**相転移** といいます。

**主な相転移** :

  * **融解（Melting）** : 固体 → 液体（融点 $T_m$）
  * **凝固（Freezing）** : 液体 → 固体
  * **蒸発（Vaporization）** : 液体 → 気体（沸点 $T_b$）
  * **凝縮（Condensation）** : 気体 → 液体
  * **昇華（Sublimation）** : 固体 → 気体（例：ドライアイス）

### 潜熱（Latent Heat）

相転移時には、温度変化を伴わずに熱の吸収・放出が起こります。この熱を**潜熱** といいます。

**融解潜熱（Latent Heat of Fusion）** $L_f$:

単位質量の物質を固体から液体に変えるのに必要な熱量

**蒸発潜熱（Latent Heat of Vaporization）** $L_v$:

単位質量の物質を液体から気体に変えるのに必要な熱量

物質 | 融点 (K) | 融解潜熱 $L_f$ (kJ/kg) | 沸点 (K) | 蒸発潜熱 $L_v$ (kJ/kg)  
---|---|---|---|---  
水（H₂O） | 273 | 334 | 373 | 2260  
鉄（Fe） | 1811 | 247 | 3134 | 6090  
アルミニウム（Al） | 933 | 397 | 2792 | 10500  
銅（Cu） | 1358 | 205 | 2835 | 4730  
  
#### 例題1.2：氷を水蒸気に変えるのに必要な熱量

**問題** : 1 kg の氷（−10°C）を水蒸気（110°C）に変えるのに必要な全熱量を計算せよ。

データ: $c_{\text{ice}} = 2.09$ kJ/(kg·K), $c_{\text{water}} = 4.18$ kJ/(kg·K), $c_{\text{steam}} = 2.01$ kJ/(kg·K), $L_f = 334$ kJ/kg, $L_v = 2260$ kJ/kg

解答を見る

**解答** :

プロセスを5段階に分けます：

  1. 氷を −10°C → 0°C に温める: $Q_1 = m c_{\text{ice}} \Delta T = 1 \times 2.09 \times 10 = 20.9$ kJ
  2. 氷を融解（0°C）: $Q_2 = m L_f = 1 \times 334 = 334$ kJ
  3. 水を 0°C → 100°C に温める: $Q_3 = m c_{\text{water}} \Delta T = 1 \times 4.18 \times 100 = 418$ kJ
  4. 水を蒸発（100°C）: $Q_4 = m L_v = 1 \times 2260 = 2260$ kJ
  5. 水蒸気を 100°C → 110°C に温める: $Q_5 = m c_{\text{steam}} \Delta T = 1 \times 2.01 \times 10 = 20.1$ kJ

全熱量: $Q_{\text{total}} = Q_1 + Q_2 + Q_3 + Q_4 + Q_5 = 20.9 + 334 + 418 + 2260 + 20.1 = 3053$ kJ

**答** : 約 3.05 MJ（メガジュール）が必要

**考察** : 蒸発潜熱（$Q_4$）が全体の約74%を占め、相転移に膨大なエネルギーが必要であることがわかります。

* * *

## 1.5 Pythonによる熱力学計算

### コード例1：温度変換と材料の熱膨張計算
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    # 温度変換関数
    def celsius_to_kelvin(celsius):
        """摂氏からケルビンへの変換"""
        return celsius + 273.15
    
    def kelvin_to_celsius(kelvin):
        """ケルビンから摂氏への変換"""
        return kelvin - 273.15
    
    def celsius_to_fahrenheit(celsius):
        """摂氏から華氏への変換"""
        return (9/5) * celsius + 32
    
    # 熱膨張計算
    def thermal_expansion(L0, alpha, delta_T):
        """線膨張による長さ変化
    
        Args:
            L0: 基準長さ (m)
            alpha: 線膨張係数 (K^-1)
            delta_T: 温度変化 (K)
    
        Returns:
            delta_L: 長さ変化 (m)
            L_final: 最終長さ (m)
        """
        delta_L = L0 * alpha * delta_T
        L_final = L0 + delta_L
        return delta_L, L_final
    
    # 材料データ
    materials = {
        'インバー合金': {'alpha': 1.2e-6, 'color': 'blue'},
        '鉄鋼': {'alpha': 12e-6, 'color': 'gray'},
        'アルミニウム': {'alpha': 23e-6, 'color': 'silver'},
        '銅': {'alpha': 17e-6, 'color': 'orange'},
        'ポリエチレン': {'alpha': 150e-6, 'color': 'green'}
    }
    
    # 初期長さと温度変化
    L0 = 1.0  # m (1メートルの棒)
    T_range = np.linspace(0, 100, 100)  # 0-100°C
    
    # プロット
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # 温度変化による長さ変化
    for material, props in materials.items():
        lengths = [thermal_expansion(L0, props['alpha'], T)[1] * 1000
                   for T in T_range]  # mm単位に変換
        ax1.plot(T_range, lengths, label=material,
                 color=props['color'], linewidth=2)
    
    ax1.set_xlabel('温度変化 ΔT (°C)', fontsize=12)
    ax1.set_ylabel('長さ (mm)', fontsize=12)
    ax1.set_title('材料の熱膨張（初期長さ1m）', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 100°Cでの膨張量比較
    delta_T_100 = 100
    expansions = []
    material_names = []
    colors = []
    
    for material, props in materials.items():
        delta_L, _ = thermal_expansion(L0, props['alpha'], delta_T_100)
        expansions.append(delta_L * 1000)  # mm単位
        material_names.append(material)
        colors.append(props['color'])
    
    ax2.barh(material_names, expansions, color=colors, alpha=0.7, edgecolor='black')
    ax2.set_xlabel('長さ変化 ΔL (mm)', fontsize=12)
    ax2.set_title(f'100°C加熱時の膨張量（初期長さ1m）', fontsize=14, fontweight='bold')
    ax2.grid(True, axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # 実用例：橋の伸縮継手
    print("=== 実用例：鋼鉄製の橋（長さ100m）の熱膨張 ===")
    bridge_length = 100  # m
    alpha_steel = 12e-6  # K^-1
    summer_winter_diff = 50  # °C（夏40°C, 冬-10°C）
    
    delta_L, _ = thermal_expansion(bridge_length, alpha_steel, summer_winter_diff)
    print(f"夏冬の温度差: {summer_winter_diff}°C")
    print(f"橋の長さ変化: {delta_L * 1000:.1f} mm = {delta_L * 100:.1f} cm")
    print(f"→ 伸縮継手で{delta_L * 100:.1f} cm以上の変位を吸収する必要がある")
    

### コード例2：理想気体の第1法則シミュレーション
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    # 物理定数
    R = 8.314  # J/(mol·K)
    
    def isothermal_work(n, T, V1, V2):
        """等温過程の仕事
    
        W = nRT ln(V2/V1)
        """
        return n * R * T * np.log(V2 / V1)
    
    def adiabatic_work(P1, V1, V2, gamma):
        """断熱過程の仕事
    
        W = (P1 V1^γ / (1-γ)) (V2^(1-γ) - V1^(1-γ))
        """
        return (P1 * V1**gamma / (1 - gamma)) * (V2**(1-gamma) - V1**(1-gamma))
    
    def isobaric_work(P, V1, V2):
        """等圧過程の仕事
    
        W = P(V2 - V1)
        """
        return P * (V2 - V1)
    
    # 初期条件
    n = 1.0  # mol
    T1 = 300  # K
    P1 = 1e5  # Pa
    V1 = n * R * T1 / P1  # m^3（理想気体の状態方程式）
    gamma = 1.4  # 二原子分子気体
    
    # 体積範囲
    V_range = np.linspace(V1, V1 * 3, 100)
    
    # 各過程でのP-V曲線
    P_isothermal = n * R * T1 / V_range
    P_adiabatic = P1 * (V1 / V_range)**gamma
    P_isobaric = P1 * np.ones_like(V_range)
    
    # プロット
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # P-V線図
    ax1 = axes[0]
    ax1.plot(V_range * 1000, P_isothermal / 1e5, 'b-', linewidth=2.5, label='等温過程')
    ax1.plot(V_range * 1000, P_adiabatic / 1e5, 'r-', linewidth=2.5, label='断熱過程')
    ax1.plot(V_range * 1000, P_isobaric / 1e5, 'g--', linewidth=2.5, label='等圧過程')
    ax1.scatter([V1 * 1000], [P1 / 1e5], color='black', s=150, zorder=5,
                marker='o', edgecolors='white', linewidths=2, label='初期状態')
    ax1.set_xlabel('体積 V (L)', fontsize=12)
    ax1.set_ylabel('圧力 P (bar)', fontsize=12)
    ax1.set_title('理想気体の準静的過程（P-V線図）', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # 仕事の計算（V1 → 2V1）
    V2 = 2 * V1
    W_isothermal = isothermal_work(n, T1, V1, V2)
    W_adiabatic = adiabatic_work(P1, V1, V2, gamma)
    W_isobaric = isobaric_work(P1, V1, V2)
    
    # 仕事の比較
    ax2 = axes[1]
    processes = ['等温', '断熱', '等圧']
    works = [W_isothermal, W_adiabatic, W_isobaric]
    colors = ['blue', 'red', 'green']
    
    bars = ax2.bar(processes, works, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    ax2.set_ylabel('系がした仕事 W (J)', fontsize=12)
    ax2.set_title(f'体積膨張の仕事（{V1*1000:.1f}L → {V2*1000:.1f}L）',
                  fontsize=14, fontweight='bold')
    ax2.grid(True, axis='y', alpha=0.3)
    
    # 数値表示
    for bar, work in zip(bars, works):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                 f'{work:.1f} J', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    # 第1法則の確認
    print("=== 熱力学第1法則の確認 ===")
    print(f"初期状態: V1 = {V1*1000:.2f} L, P1 = {P1/1e5:.2f} bar, T1 = {T1} K")
    print(f"最終状態: V2 = {V2*1000:.2f} L\n")
    
    print("【等温過程】")
    print(f"  仕事 W = {W_isothermal:.2f} J")
    print(f"  内部エネルギー変化 ΔU = 0 (等温)")
    print(f"  吸収熱 Q = ΔU + W = {W_isothermal:.2f} J")
    print(f"  → 系は仕事をするのと同じだけの熱を吸収\n")
    
    print("【断熱過程】")
    print(f"  仕事 W = {W_adiabatic:.2f} J")
    print(f"  吸収熱 Q = 0 (断熱)")
    print(f"  内部エネルギー変化 ΔU = Q - W = {-W_adiabatic:.2f} J")
    print(f"  → 内部エネルギーが減少し、温度が下がる\n")
    
    print("【等圧過程】")
    print(f"  仕事 W = {W_isobaric:.2f} J")
    print(f"  内部エネルギー変化 ΔU = nCvΔT (温度上昇)")
    print(f"  吸収熱 Q = ΔU + W")
    print(f"  → エンタルピー変化 ΔH = Q")
    

### コード例3：材料の比熱測定シミュレーション
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    # 材料の比熱データ（J/(g·K)）
    materials_data = {
        '銅': {'c': 0.385, 'color': 'orange'},
        '鉄': {'c': 0.449, 'color': 'gray'},
        'アルミニウム': {'c': 0.900, 'color': 'silver'},
        '水': {'c': 4.18, 'color': 'blue'},
        'エタノール': {'c': 2.44, 'color': 'green'}
    }
    
    def heat_required(mass, c, delta_T):
        """温度変化に必要な熱量
    
        Q = m c ΔT
    
        Args:
            mass: 質量 (g)
            c: 比熱 (J/(g·K))
            delta_T: 温度変化 (K)
    
        Returns:
            Q: 必要な熱量 (J)
        """
        return mass * c * delta_T
    
    def temperature_change(Q, mass, c):
        """与えられた熱量による温度変化
    
        ΔT = Q / (m c)
        """
        return Q / (mass * c)
    
    # シミュレーション：100gの各材料を20°C→100°C に温める
    mass = 100  # g
    T_initial = 20  # °C
    T_final = 100  # °C
    delta_T = T_final - T_initial
    
    # 必要な熱量を計算
    heat_requirements = {}
    for material, props in materials_data.items():
        Q = heat_required(mass, props['c'], delta_T)
        heat_requirements[material] = Q
    
    # プロット1：必要な熱量の比較
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    ax1 = axes[0, 0]
    materials = list(heat_requirements.keys())
    heats = list(heat_requirements.values())
    colors = [materials_data[m]['color'] for m in materials]
    
    bars = ax1.bar(materials, heats, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    ax1.set_ylabel('必要な熱量 (J)', fontsize=11)
    ax1.set_title(f'100gの材料を{T_initial}°C→{T_final}°Cに温めるのに必要な熱量',
                  fontsize=12, fontweight='bold')
    ax1.grid(True, axis='y', alpha=0.3)
    ax1.tick_params(axis='x', rotation=15)
    
    for bar, heat in zip(bars, heats):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                 f'{heat:.0f} J', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # プロット2：温度上昇曲線（同じ熱量1000Jを与える）
    ax2 = axes[0, 1]
    Q_input = 1000  # J
    
    for material, props in materials_data.items():
        delta_T_achieved = temperature_change(Q_input, mass, props['c'])
        T_range = np.array([T_initial, T_initial + delta_T_achieved])
        Q_range = np.array([0, Q_input])
    
        ax2.plot(Q_range, T_range, linewidth=2.5, marker='o', markersize=8,
                 label=material, color=props['color'])
    
    ax2.set_xlabel('投入熱量 (J)', fontsize=11)
    ax2.set_ylabel('温度 (°C)', fontsize=11)
    ax2.set_title(f'100gの材料に{Q_input}Jの熱を与えたときの温度上昇',
                  fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # プロット3：加熱時間による温度変化（一定の加熱速度）
    ax3 = axes[1, 0]
    heating_power = 100  # W（100 J/s）
    time_range = np.linspace(0, 50, 100)  # 0-50秒
    
    for material, props in materials_data.items():
        Q_vs_time = heating_power * time_range
        T_vs_time = T_initial + temperature_change(Q_vs_time, mass, props['c'])
    
        ax3.plot(time_range, T_vs_time, linewidth=2.5,
                 label=material, color=props['color'])
    
    ax3.set_xlabel('時間 (s)', fontsize=11)
    ax3.set_ylabel('温度 (°C)', fontsize=11)
    ax3.set_title(f'100Wで加熱したときの温度変化（質量100g）',
                  fontsize=12, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    # プロット4：比熱の比較
    ax4 = axes[1, 1]
    specific_heats = [materials_data[m]['c'] for m in materials]
    colors_sh = [materials_data[m]['color'] for m in materials]
    
    bars = ax4.barh(materials, specific_heats, color=colors_sh, alpha=0.7,
                    edgecolor='black', linewidth=2)
    ax4.set_xlabel('比熱 c (J/(g·K))', fontsize=11)
    ax4.set_title('材料の比熱比較', fontsize=12, fontweight='bold')
    ax4.grid(True, axis='x', alpha=0.3)
    
    for bar, c_val in zip(bars, specific_heats):
        width = bar.get_width()
        ax4.text(width, bar.get_y() + bar.get_height()/2.,
                 f'{c_val:.3f}', ha='left', va='center', fontsize=9,
                 fontweight='bold', color='black')
    
    plt.tight_layout()
    plt.show()
    
    # 実用例の計算
    print("=== 実用例：材料選択における比熱の重要性 ===\n")
    print("【冷却材としての水】")
    water_mass = 10000  # g (10 kg)
    delta_T_cooling = 20  # K（80°C → 60°C）
    Q_removed = heat_required(water_mass, materials_data['水']['c'], delta_T_cooling)
    print(f"10 kgの水が80°C→60°Cに冷却される際に放出する熱量:")
    print(f"Q = {Q_removed / 1000:.1f} kJ")
    print(f"→ 高比熱により大量の熱を吸収/放出可能（冷却材に最適）\n")
    
    print("【ヒートシンク材料としての銅】")
    cu_mass = 100  # g
    Q_heat = 1000  # J（CPUが発する熱）
    delta_T_cu = temperature_change(Q_heat, cu_mass, materials_data['銅']['c'])
    print(f"100gの銅製ヒートシンクが1000Jの熱を吸収したときの温度上昇:")
    print(f"ΔT = {delta_T_cu:.1f}°C")
    print(f"→ 低比熱により温度上昇しやすいが、高熱伝導率で熱を素早く拡散")
    

### コード例4：相転移の熱量計算
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    # 水の熱力学データ
    c_ice = 2.09      # kJ/(kg·K)
    c_water = 4.18    # kJ/(kg·K)
    c_steam = 2.01    # kJ/(kg·K)
    L_fusion = 334    # kJ/kg
    L_vaporization = 2260  # kJ/kg
    
    def heating_curve(mass, T_start, T_end):
        """加熱曲線のシミュレーション
    
        Args:
            mass: 質量 (kg)
            T_start: 開始温度 (°C)
            T_end: 終了温度 (°C)
    
        Returns:
            Q_cumulative: 累積熱量 (kJ)
            T_curve: 温度履歴 (°C)
            labels: 各段階のラベル
        """
        Q_cumulative = [0]
        T_curve = [T_start]
        labels = []
    
        # Stage 1: 氷を−20°C → 0°Cに温める
        if T_start < 0:
            Q1 = mass * c_ice * (0 - T_start)
            Q_cumulative.append(Q_cumulative[-1] + Q1)
            T_curve.append(0)
            labels.append(f"氷加熱 ({T_start}→0°C)")
    
        # Stage 2: 氷を融解（0°C）
        Q2 = mass * L_fusion
        Q_cumulative.append(Q_cumulative[-1] + Q2)
        T_curve.append(0)
        labels.append("融解（氷→水）")
    
        # Stage 3: 水を0°C → 100°Cに温める
        Q3 = mass * c_water * (100 - 0)
        Q_cumulative.append(Q_cumulative[-1] + Q3)
        T_curve.append(100)
        labels.append("水加熱（0→100°C）")
    
        # Stage 4: 水を蒸発（100°C）
        Q4 = mass * L_vaporization
        Q_cumulative.append(Q_cumulative[-1] + Q4)
        T_curve.append(100)
        labels.append("蒸発（水→水蒸気）")
    
        # Stage 5: 水蒸気を100°C → T_endに温める
        if T_end > 100:
            Q5 = mass * c_steam * (T_end - 100)
            Q_cumulative.append(Q_cumulative[-1] + Q5)
            T_curve.append(T_end)
            labels.append(f"水蒸気加熱（100→{T_end}°C）")
    
        return np.array(Q_cumulative), np.array(T_curve), labels
    
    # シミュレーション
    mass = 1.0  # kg
    T_start = -20  # °C
    T_end = 120  # °C
    
    Q_cumulative, T_curve, stage_labels = heating_curve(mass, T_start, T_end)
    
    # プロット
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 加熱曲線
    ax1 = axes[0]
    ax1.plot(Q_cumulative, T_curve, 'b-', linewidth=3, marker='o', markersize=8)
    
    # 相転移点を強調
    ax1.axhline(0, color='cyan', linestyle='--', linewidth=2, alpha=0.7, label='融点（0°C）')
    ax1.axhline(100, color='red', linestyle='--', linewidth=2, alpha=0.7, label='沸点（100°C）')
    
    # 領域を色分け
    ax1.axhspan(-20, 0, alpha=0.1, color='cyan', label='固相（氷）')
    ax1.axhspan(0, 100, alpha=0.1, color='blue', label='液相（水）')
    ax1.axhspan(100, 120, alpha=0.1, color='red', label='気相（水蒸気）')
    
    ax1.set_xlabel('累積投入熱量 Q (kJ)', fontsize=12)
    ax1.set_ylabel('温度 (°C)', fontsize=12)
    ax1.set_title(f'{mass} kgの水の加熱曲線', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # 各段階の熱量内訳
    ax2 = axes[1]
    Q_stages = np.diff(Q_cumulative)
    colors = ['lightblue', 'cyan', 'blue', 'red', 'orange']
    
    bars = ax2.bar(range(len(Q_stages)), Q_stages, color=colors, alpha=0.7,
                   edgecolor='black', linewidth=2)
    ax2.set_xticks(range(len(Q_stages)))
    ax2.set_xticklabels([f'段階{i+1}' for i in range(len(Q_stages))], rotation=0)
    ax2.set_ylabel('必要な熱量 (kJ)', fontsize=12)
    ax2.set_title(f'各段階の熱量内訳（質量{mass} kg）', fontsize=14, fontweight='bold')
    ax2.grid(True, axis='y', alpha=0.3)
    
    # 数値表示
    for i, (bar, Q_val, label) in enumerate(zip(bars, Q_stages, stage_labels)):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                 f'{Q_val:.0f} kJ\n({Q_val/Q_cumulative[-1]*100:.1f}%)',
                 ha='center', va='bottom', fontsize=9, fontweight='bold')
        # ラベルを下に表示
        ax2.text(i, -100, label, ha='center', va='top', fontsize=8, rotation=0, wrap=True)
    
    ax2.set_ylim(bottom=-150)
    
    plt.tight_layout()
    plt.show()
    
    # 詳細な出力
    print("=== 水の加熱プロセス詳細（質量1kg）===\n")
    for i, (Q_start, Q_end, T, label) in enumerate(zip(Q_cumulative[:-1], Q_cumulative[1:],
                                                         T_curve[1:], stage_labels)):
        Q_stage = Q_end - Q_start
        print(f"【段階{i+1}】{label}")
        print(f"  投入熱量: {Q_stage:.1f} kJ ({Q_stage/Q_cumulative[-1]*100:.1f}%)")
        print(f"  累積熱量: {Q_end:.1f} kJ")
        print(f"  温度: {T:.1f}°C\n")
    
    print(f"合計必要熱量: {Q_cumulative[-1]:.1f} kJ = {Q_cumulative[-1]/1000:.2f} MJ\n")
    
    print("【考察】")
    print(f"・蒸発潜熱（{L_vaporization} kJ/kg）が全体の{Q_stages[3]/Q_cumulative[-1]*100:.1f}%を占める")
    print(f"・融解潜熱（{L_fusion} kJ/kg）は全体の{Q_stages[1]/Q_cumulative[-1]*100:.1f}%")
    print("・相転移に膨大なエネルギーが必要 → 蒸発冷却の原理、蒸留プロセスなど")
    

* * *

## 演習問題

### Easy（基礎確認）

**Q1** : 熱力学の第0法則が定義するものは何ですか？

**正解** : 温度

**解説** :

熱力学の第0法則は、「物体AとBが熱平衡、BとCが熱平衡ならば、AとCも熱平衡にある」という推移律を述べています。この法則により、「温度」という状態量を定義でき、温度計による測定が可能になります。

**Q2** : 熱力学第1法則の式 $\Delta U = Q - W$ において、$Q > 0$, $W > 0$ のとき、系は熱を吸収していますか、それとも放出していますか？

**正解** : 吸収している

**解説** :

符号規約により：

  * $Q > 0$: 系が熱を**吸収**
  * $W > 0$: 系が外部に**仕事をする** （例：気体の膨張）

この場合、系は熱を吸収し、その一部を仕事として外部に出力し、残りが内部エネルギーの増加に寄与します。

**Q3** : 比熱が最も大きい物質は次のうちどれですか？ (a) 銅 (b) 鉄 (c) 水 (d) アルミニウム

**正解** : (c) 水

**解説** :

  * 銅: 0.385 J/(g·K)
  * 鉄: 0.449 J/(g·K)
  * 水: 4.18 J/(g·K) ← 最大
  * アルミニウム: 0.900 J/(g·K)

水は液体の中でも特に高い比熱を持ち、これが地球の気候調節や生命活動において重要な役割を果たしています。

### Medium（応用）

**Q4** : 500 gの鉄（比熱0.449 J/(g·K)）を20°Cから100°Cまで温めるのに必要な熱量を計算してください。

**正解** : 17964 J ≈ 18.0 kJ

**計算** :

$$Q = mc\Delta T = (500 \text{ g}) \times (0.449 \text{ J/(g·K)}) \times (100 - 20) \text{ K}$$

$$Q = 500 \times 0.449 \times 80 = 17964 \text{ J} = 17.96 \text{ kJ}$$

**考察** :

鉄は比熱が比較的小さいため、同じ質量の水（4.18 J/(g·K)）に比べて約9分の1の熱量で同じ温度変化が達成できます。これは調理器具（鉄のフライパンなど）が早く温まる理由の一つです。

**Q5** : 理想気体が等温膨張するとき、内部エネルギー変化 $\Delta U$ はいくらですか？その理由も説明してください。

**正解** : $\Delta U = 0$

**解説** :

理想気体の内部エネルギーは温度のみの関数です：$U = U(T)$

等温過程では温度が一定（$T = \text{const}$）なので、内部エネルギーも変化しません：$\Delta U = 0$

第1法則 $\Delta U = Q - W$ より：

$$0 = Q - W \quad \Rightarrow \quad Q = W$$

つまり、系が吸収した熱は全て仕事として外部に出力されます。

### Hard（発展）

**Q6** : 100 gの氷（−10°C）を水（50°C）に変えるのに必要な全熱量を計算してください。データ: $c_{\text{ice}} = 2.09$ J/(g·K), $c_{\text{water}} = 4.18$ J/(g·K), $L_f = 334$ J/g

**正解** : 約 56.3 kJ

**計算** :

プロセスを3段階に分けます：

  1. **氷を−10°C → 0°Cに温める** : 

$$Q_1 = mc_{\text{ice}}\Delta T = (100)(2.09)(10) = 2090 \text{ J}$$

  2. **氷を融解（0°C）** : 

$$Q_2 = mL_f = (100)(334) = 33400 \text{ J}$$

  3. **水を0°C → 50°Cに温める** : 

$$Q_3 = mc_{\text{water}}\Delta T = (100)(4.18)(50) = 20900 \text{ J}$$

全熱量:

$$Q_{\text{total}} = Q_1 + Q_2 + Q_3 = 2090 + 33400 + 20900 = 56390 \text{ J} \approx 56.4 \text{ kJ}$$

**考察** :

  * 融解潜熱（$Q_2$）が全体の約59%を占める
  * 相転移には温度変化を伴わないが、大量のエネルギーが必要
  * この性質を利用して、氷は効果的な冷却剤として使用される

**Q7** : アルミニウム製の棒（長さ10 m、線膨張係数23×10⁻⁶ K⁻¹）が夏（40°C）と冬（−10°C）で何cm伸縮するか計算してください。建築設計においてこの値が重要な理由も述べてください。

**正解** : 約 1.15 cm

**計算** :

$$\Delta L = L_0 \alpha \Delta T$$

$$\Delta L = (10 \text{ m}) \times (23 \times 10^{-6} \text{ K}^{-1}) \times (40 - (-10)) \text{ K}$$

$$\Delta L = 10 \times 23 \times 10^{-6} \times 50 = 0.0115 \text{ m} = 1.15 \text{ cm}$$

**建築設計における重要性** :

  * **伸縮継手（Expansion Joint）** : 橋や建物に熱膨張を吸収する継手が必須
  * **材料の組み合わせ** : 異なる材料を接合する際、熱膨張係数の差による応力を考慮
  * **レール敷設** : 鉄道レールには適切な隙間を設けて熱膨張に対応
  * **精密機器** : インバー合金など低膨張材料を使用して寸法安定性を確保

**Q8** : Dulong-Petitの法則によると、金属の原子あたりのモル熱容量は約 $3R$ です。この値を導出し、低温でこの法則が成り立たない理由を量子論の観点から説明してください。

**解答** :

**古典論による導出** :

固体中の原子は3次元調和振動子とみなせます。各原子は3つの並進自由度を持ち、エネルギー等分配の定理により：

  * 運動エネルギー（3方向）: $3 \times \frac{1}{2}k_B T$
  * ポテンシャルエネルギー（3方向）: $3 \times \frac{1}{2}k_B T$

1原子あたりの平均エネルギー: $E = 3k_B T$

1 molの場合: $U = N_A \times 3k_B T = 3RT$

モル熱容量: $C_V = \frac{\partial U}{\partial T} = 3R \approx 25 \text{ J/(mol·K)}$

**低温での破綻（量子論）** :

量子論では、振動エネルギーは連続ではなく離散的（量子化）：

$$E_n = \left(n + \frac{1}{2}\right)\hbar\omega$$

低温（$k_B T \ll \hbar\omega$）では：

  * 多くの振動モードが励起されない（基底状態に凍結）
  * 熱容量が $T^3$ に比例して減少（Debye T³則）
  * 絶対零度で $C_V \to 0$（熱力学第3法則と整合）

これはDebyeモデルにより正確に記述されます：

$$C_V = 9Nk_B \left(\frac{T}{\Theta_D}\right)^3 \int_0^{\Theta_D/T} \frac{x^4 e^x}{(e^x - 1)^2} dx$$

ここで $\Theta_D$ はDebye温度です。

* * *

## 本章のまとめ

### 学んだこと

  1. **熱力学の第0法則**
     * 熱平衡の推移律により温度の概念を定義
     * 温度測定の原理と温度スケール（K, °C, °F）
     * 材料の熱膨張と線膨張係数
  2. **熱力学の第1法則**
     * エネルギー保存則: $\Delta U = Q - W$
     * 内部エネルギー、熱、仕事の関係
     * 等温、断熱、等圧過程における挙動
  3. **熱容量と比熱**
     * 定容熱容量 $C_V$ と定圧熱容量 $C_P$
     * Dulong-Petitの法則とDebyeモデル
     * 材料による比熱の違いと応用
  4. **相転移と潜熱**
     * 融解・蒸発における潜熱の役割
     * 加熱曲線と各段階のエネルギー収支
     * 材料プロセスにおける相転移の重要性

### 重要なポイント

  * 温度は熱平衡を特徴づける状態量であり、第0法則により定義される
  * 第1法則は永久機関の不可能性を示し、エネルギー収支計算の基礎となる
  * 材料の比熱は熱管理や熱処理プロセスにおいて重要な設計パラメータ
  * 相転移では大量の潜熱が必要であり、これを利用した冷却・蓄熱技術がある
  * 熱膨張は材料選択や設計において考慮すべき重要な要素

### 次の章へ

第2章では、**熱力学の第2法則とエントロピー** を学びます：

  * エントロピーの定義と物理的意味
  * 可逆過程と不可逆過程
  * Carnotサイクルと熱機関の効率
  * 材料のエントロピーと統計力学的解釈
  * 自由エネルギーと化学平衡
