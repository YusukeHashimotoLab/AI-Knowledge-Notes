---
title: "Chapter 3: 金属材料の強化機構"
chapter_title: "Chapter 3: 金属材料の強化機構"
subtitle: 固溶強化、析出強化、加工硬化、結晶粒微細化、Hall-Petch則、Orowan機構
---

# Chapter 3: 金属材料の強化機構

固溶強化、析出強化、加工硬化、結晶粒微細化、Hall-Petch則、Orowan機構

## 3.1 金属材料の強化機構概要

金属材料の強度を向上させるには、転位の運動を妨げることが基本原理です。主な強化機構として、(1)固溶強化、(2)析出強化、(3)加工硬化、(4)結晶粒微細化、(5)分散強化があります。
    
    
    ```mermaid
    flowchart TD; A[金属材料の強化機構]-->B[固溶強化]; A-->C[析出強化]; A-->D[加工硬化]; A-->E[結晶粒微細化]; A-->F[分散強化]; B-->B1[置換型固溶強化]; B-->B2[侵入型固溶強化]; C-->C1[Orowan機構]; C-->C2[せん断機構]; D-->D1[転位密度増加]; E-->E1[Hall-Petch則]
    ```

### 3.1.1 Hall-Petch則

**Hall-Petch則：**

$$\sigma_y = \sigma_0 + k_y d^{-1/2}$$

ここで、$\sigma_y$: 降伏応力、$\sigma_0$: 摩擦応力、$k_y$: Hall-Petch定数、$d$: 結晶粒径

#### コード例1: Hall-Petch則による降伏応力計算
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    def hall_petch_strength(d, sigma_0, k_y):
        """Hall-Petch則で降伏応力計算
        Parameters: d(μm), sigma_0(MPa), k_y(MPa·μm^0.5)
        Returns: sigma_y(MPa)"""
        return sigma_0 + k_y * d**(-0.5)
    
    # 鋼のパラメータ
    sigma_0_steel = 70  # MPa
    k_y_steel = 0.74    # MPa·mm^0.5 = 740 MPa·μm^0.5
    d_range = np.linspace(1, 100, 100)  # μm
    sigma_y_steel = hall_petch_strength(d_range, sigma_0_steel, k_y_steel * 1000)
    
    # アルミニウムのパラメータ
    sigma_0_al = 20
    k_y_al = 0.11 * 1000
    sigma_y_al = hall_petch_strength(d_range, sigma_0_al, k_y_al)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(d_range, sigma_y_steel, 'b-', linewidth=2.5, label='鋼')
    ax1.plot(d_range, sigma_y_al, 'r-', linewidth=2.5, label='Al')
    ax1.set_xlabel('結晶粒径 d (μm)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('降伏応力 σy (MPa)', fontsize=12, fontweight='bold')
    ax1.set_title('Hall-Petch則：降伏応力 vs 結晶粒径', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11); ax1.grid(True, alpha=0.3)
    
    d_inv_sqrt = d_range**(-0.5)
    ax2.plot(d_inv_sqrt, sigma_y_steel, 'bo-', linewidth=2, markersize=4, label='鋼')
    ax2.plot(d_inv_sqrt, sigma_y_al, 'ro-', linewidth=2, markersize=4, label='Al')
    ax2.set_xlabel('d^(-1/2) (μm^(-1/2))', fontsize=12, fontweight='bold')
    ax2.set_ylabel('降伏応力 σy (MPa)', fontsize=12, fontweight='bold')
    ax2.set_title('Hall-Petch プロット（直線関係）', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11); ax2.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig('hall_petch_law.png', dpi=300, bbox_inches='tight'); plt.show()
    print(f"鋼（d=10μm）: σy = {hall_petch_strength(10, sigma_0_steel, k_y_steel*1000):.1f} MPa")
    print(f"Al（d=10μm）: σy = {hall_petch_strength(10, sigma_0_al, k_y_al):.1f} MPa")

## 3.2 固溶強化

固溶強化は、溶質原子を母相に固溶させ、格子歪みや弾性率の違いにより転位運動を妨げる機構です。

### 3.2.1 固溶強化の機構

**固溶強化による強度増加：**

$$\Delta\sigma_{ss} = G \epsilon^{3/2} c^{1/2}$$

$G$: せん断弾性率、$\epsilon$: ミスフィット歪み、$c$: 溶質濃度

#### コード例2: 固溶強化効果の計算
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    def solid_solution_strengthening(c, G, epsilon, A=1.0):
        """固溶強化による強度増加計算
        Parameters: c(at%), G(GPa), epsilon(無次元), A(定数)
        Returns: Delta_sigma(MPa)"""
        c_fraction = c / 100
        return A * G * 1000 * epsilon**(3/2) * c_fraction**(1/2)
    
    # Cu-Zn系（黄銅）
    G_Cu = 48  # GPa
    epsilon_Zn_in_Cu = 0.04  # Znの格子ミスフィット
    c_Zn_range = np.linspace(0, 40, 100)
    Delta_sigma_CuZn = solid_solution_strengthening(c_Zn_range, G_Cu, epsilon_Zn_in_Cu, A=200)
    
    # Al-Mg系
    G_Al = 26
    epsilon_Mg_in_Al = 0.12
    c_Mg_range = np.linspace(0, 6, 100)
    Delta_sigma_AlMg = solid_solution_strengthening(c_Mg_range, G_Al, epsilon_Mg_in_Al, A=150)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(c_Zn_range, Delta_sigma_CuZn, 'b-', linewidth=2.5, label='Cu-Zn（黄銅）')
    ax.plot(c_Mg_range, Delta_sigma_AlMg, 'r-', linewidth=2.5, label='Al-Mg')
    ax.set_xlabel('溶質濃度 (at%)', fontsize=12, fontweight='bold')
    ax.set_ylabel('固溶強化による強度増加 Δσ (MPa)', fontsize=12, fontweight='bold')
    ax.set_title('固溶強化効果：濃度依存性', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11); ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig('solid_solution_strengthening.png', dpi=300); plt.show()
    print(f"Cu-30Zn: Δσ = {solid_solution_strengthening(30, G_Cu, epsilon_Zn_in_Cu, A=200):.1f} MPa")
    print(f"Al-5Mg: Δσ = {solid_solution_strengthening(5, G_Al, epsilon_Mg_in_Al, A=150):.1f} MPa")

## 3.3 析出強化

析出強化は、微細な析出粒子を分散させ、転位運動を妨げる最も効果的な強化機構です。Orowan機構とせん断機構があります。

### 3.3.1 Orowan機構

**Orowan応力：**

$$\tau_{Orowan} = \frac{Gb}{L}$$

$G$: せん断弾性率、$b$: バーガースベクトル、$L$: 粒子間隔

#### コード例3: Orowan応力計算
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    def orowan_stress(G, b, L):
        """Orowan応力計算
        Parameters: G(GPa), b(nm), L(nm)
        Returns: tau(MPa)"""
        return (G * 1000 * b) / L
    
    def particle_spacing(r, f):
        """粒子間隔計算（体積分率から）
        Parameters: r(nm粒子半径), f(体積分率)
        Returns: L(nm)"""
        return r * np.sqrt(2 * np.pi / (3 * f))
    
    # Al合金のパラメータ
    G_Al = 26  # GPa
    b_Al = 0.286  # nm（Al のバーガースベクトル）
    r_range = np.linspace(5, 50, 100)  # nm
    
    # 異なる体積分率での計算
    volume_fractions = [0.01, 0.02, 0.05, 0.10]
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for f in volume_fractions:
        L_values = particle_spacing(r_range, f)
        tau_values = orowan_stress(G_Al, b_Al, L_values)
        ax.plot(r_range, tau_values, linewidth=2.5, label=f'f = {f*100:.0f}%')
    
    ax.set_xlabel('析出粒子半径 r (nm)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Orowan応力 τ (MPa)', fontsize=12, fontweight='bold')
    ax.set_title('Orowan機構：析出強化効果', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11); ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig('orowan_strengthening.png', dpi=300); plt.show()
    
    # 具体例：Al-Cu合金（θ'析出物）
    r_theta_prime = 20  # nm
    f_theta_prime = 0.03
    L = particle_spacing(r_theta_prime, f_theta_prime)
    tau = orowan_stress(G_Al, b_Al, L)
    print(f"Al-Cu合金（θ'析出）: r={r_theta_prime}nm, f={f_theta_prime*100}%")
    print(f"  粒子間隔 L = {L:.1f} nm")
    print(f"  Orowan応力 τ = {tau:.1f} MPa")

## 3.4 加工硬化

加工硬化（work hardening）は、塑性変形により転位密度が増加し、転位同士の相互作用で強度が上昇する現象です。

**加工硬化による強度増加：**

$$\Delta\sigma_{wh} = \alpha G b \sqrt{\rho}$$

$\alpha$: 定数（約0.3）、$\rho$: 転位密度（m^-2）

#### コード例4: 加工硬化曲線のフィッティング
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.optimize import curve_fit
    
    def hollomon_equation(epsilon, K, n):
        """Hollomon式（加工硬化則）
        σ = K * ε^n
        Parameters: epsilon(ひずみ), K(強度係数MPa), n(加工硬化指数)
        Returns: sigma(MPa)"""
        return K * epsilon**n
    
    # 実験データ（例：低炭素鋼の応力-ひずみ曲線）
    epsilon_exp = np.array([0.002, 0.005, 0.01, 0.02, 0.05, 0.10, 0.15, 0.20])
    sigma_exp = np.array([250, 280, 310, 350, 420, 480, 520, 550])
    
    # Hollomon式でフィッティング
    popt, pcov = curve_fit(hollomon_equation, epsilon_exp, sigma_exp)
    K_fit, n_fit = popt
    
    epsilon_fit = np.linspace(0.001, 0.25, 200)
    sigma_fit = hollomon_equation(epsilon_fit, K_fit, n_fit)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epsilon_exp, sigma_exp, 'ro', markersize=10, label='実験データ')
    ax.plot(epsilon_fit, sigma_fit, 'b-', linewidth=2.5, label=f'Hollomon式フィット\nK={K_fit:.1f}MPa, n={n_fit:.3f}')
    ax.set_xlabel('真ひずみ ε', fontsize=12, fontweight='bold')
    ax.set_ylabel('真応力 σ (MPa)', fontsize=12, fontweight='bold')
    ax.set_title('加工硬化曲線（応力-ひずみ関係）', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11); ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig('work_hardening_curve.png', dpi=300); plt.show()
    
    print(f"Hollomon式フィッティング結果:")
    print(f"  強度係数 K = {K_fit:.1f} MPa")
    print(f"  加工硬化指数 n = {n_fit:.3f}")
    print(f"\n加工硬化指数 n の意味:")
    print(f"  n ≈ 0.1-0.2: 低加工硬化性（軟鋼）")
    print(f"  n ≈ 0.2-0.3: 中程度（普通鋼）")
    print(f"  n ≈ 0.3-0.5: 高加工硬化性（ステンレス鋼、銅）")

### 演習問題

#### 演習 3.1

Easy

**問題：** 鋼の結晶粒径が5μmから20μmに変化したとき、Hall-Petch則による降伏応力の変化を計算してください。σ₀=70MPa、k_y=740MPa·μm^0.5とします。

解答を表示
    
    
    import numpy as np
    
    sigma_0 = 70  # MPa
    k_y = 740     # MPa·μm^0.5
    d1, d2 = 5, 20  # μm
    
    sigma_y1 = sigma_0 + k_y * d1**(-0.5)
    sigma_y2 = sigma_0 + k_y * d2**(-0.5)
    delta_sigma = sigma_y1 - sigma_y2
    
    print(f"結晶粒径 d = {d1} μm: σy = {sigma_y1:.1f} MPa")
    print(f"結晶粒径 d = {d2} μm: σy = {sigma_y2:.1f} MPa")
    print(f"降伏応力の変化 Δσy = {delta_sigma:.1f} MPa")
    print(f"\n結晶粒径が{d1}μmから{d2}μmに粗大化すると、")
    print(f"降伏応力は{delta_sigma:.1f}MPa低下します。")
    print("結晶粒微細化は効果的な強化機構です。")

#### 演習 3.2

Medium

**問題：** Al合金（G=26GPa、b=0.286nm）において、半径15nmの析出粒子が体積分率5%で分散しています。Orowan機構による強化効果を計算してください。

解答を表示
    
    
    import numpy as np
    
    G = 26  # GPa
    b = 0.286  # nm
    r = 15  # nm
    f = 0.05  # 体積分率
    
    L = r * np.sqrt(2 * np.pi / (3 * f))
    tau_orowan = (G * 1000 * b) / L
    sigma_orowan = tau_orowan * 3.06  # Taylor因子（多結晶）
    
    print(f"与えられた値:")
    print(f"  せん断弾性率 G = {G} GPa")
    print(f"  バーガースベクトル b = {b} nm")
    print(f"  析出粒子半径 r = {r} nm")
    print(f"  体積分率 f = {f*100}%")
    print(f"\n粒子間隔の計算:")
    print(f"  L = r√(2π/3f) = {r}√(2π/(3×{f}))")
    print(f"    = {L:.1f} nm")
    print(f"\nOrowan応力の計算:")
    print(f"  τ = Gb/L = ({G}×10³×{b})/{L:.1f}")
    print(f"    = {tau_orowan:.1f} MPa")
    print(f"\n引張強度への換算（Taylor因子≈3.06）:")
    print(f"  Δσ = 3.06 × τ = {sigma_orowan:.1f} MPa")
    print(f"\n結論: Orowan機構により約{sigma_orowan:.0f}MPaの強化効果が得られます。")

#### 演習 3.3

Hard

**問題：** Al-4%Cu合金において、結晶粒微細化（Hall-Petch）、固溶強化、析出強化の3つの機構が同時に働く場合、総合的な降伏応力を計算してください。パラメータ：σ₀=20MPa、k_y=110MPa·μm^0.5、d=10μm、Cu固溶強化=50MPa、析出強化=200MPa。

解答を表示
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    # 基本パラメータ
    sigma_0 = 20  # MPa（純Alの摩擦応力）
    k_y = 110     # MPa·μm^0.5
    d = 10        # μm
    Delta_sigma_ss = 50   # MPa（固溶強化）
    Delta_sigma_ppt = 200 # MPa（析出強化）
    
    # 各強化機構の寄与
    sigma_HP = k_y * d**(-0.5)
    sigma_total = sigma_0 + sigma_HP + Delta_sigma_ss + Delta_sigma_ppt
    
    print("=" * 70)
    print("Al-4%Cu合金の総合強度計算")
    print("=" * 70)
    print(f"\n(1) 基本強度（純Al）: σ₀ = {sigma_0} MPa")
    print(f"\n(2) Hall-Petch強化（結晶粒微細化）:")
    print(f"    σ_HP = k_y × d^(-1/2)")
    print(f"         = {k_y} × {d}^(-1/2)")
    print(f"         = {sigma_HP:.1f} MPa")
    print(f"\n(3) 固溶強化（Cuの固溶）: Δσ_ss = {Delta_sigma_ss} MPa")
    print(f"\n(4) 析出強化（θ'相析出）: Δσ_ppt = {Delta_sigma_ppt} MPa")
    print(f"\n(5) 総合降伏応力（線形加算）:")
    print(f"    σ_y = σ₀ + σ_HP + Δσ_ss + Δσ_ppt")
    print(f"        = {sigma_0} + {sigma_HP:.1f} + {Delta_sigma_ss} + {Delta_sigma_ppt}")
    print(f"        = {sigma_total:.1f} MPa")
    
    # 各機構の寄与率
    contributions = [sigma_0, sigma_HP, Delta_sigma_ss, Delta_sigma_ppt]
    labels = ['基本強度', 'Hall-Petch', '固溶強化', '析出強化']
    colors = ['#lightgray', '#4ECDC4', '#FF6B6B', '#FFA07A']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # 左図：積み上げ棒グラフ
    bottom = 0
    for contrib, label, color in zip(contributions, labels, colors):
        ax1.bar('Al-4Cu合金', contrib, bottom=bottom, label=label, color=color, edgecolor='black', linewidth=1.5)
        bottom += contrib
    ax1.set_ylabel('降伏応力 (MPa)', fontsize=12, fontweight='bold')
    ax1.set_title('各強化機構の寄与（積み上げ）', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_ylim(0, sigma_total * 1.1)
    
    # 右図：円グラフ
    wedges, texts, autotexts = ax2.pie(contributions, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90, textprops={'fontsize':11,'fontweight':'bold'})
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    ax2.set_title('各強化機構の寄与率', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('al_cu_combined_strengthening.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n" + "=" * 70)
    print("【結論】")
    print("=" * 70)
    print(f"Al-4%Cu合金（結晶粒径{d}μm、時効処理後）の降伏応力:")
    print(f"  σ_y = {sigma_total:.0f} MPa")
    print(f"\n最も効果的な強化機構: 析出強化（{Delta_sigma_ppt/sigma_total*100:.1f}%寄与）")
    print(f"次に効果的: 固溶強化（{Delta_sigma_ss/sigma_total*100:.1f}%寄与）")
    print("\n実用的なアルミニウム合金では、複数の強化機構を")
    print("組み合わせることで高強度化を実現しています。")
    print("=" * 70)

### 学習目標の確認

#### レベル1: 基本理解

  * Hall-Petch則の物理的意味を説明できる
  * 固溶強化、析出強化、加工硬化の違いを理解している
  * Orowan機構の原理を説明できる

#### レベル2: 実践スキル

  * Hall-Petch則で降伏応力を計算できる
  * 固溶強化効果を定量的に評価できる
  * Orowan応力を計算し、析出強化を設計できる
  * 加工硬化曲線をフィッティングできる
  * 複数の強化機構を組み合わせて総合強度を予測できる

#### レベル3: 応用力

  * 目標強度を達成するための合金設計ができる
  * 熱処理条件を最適化して析出強化を制御できる
  * 実験データから強化機構を解析できる
  * 材料の強度-延性バランスを最適化できる

## 参考文献

  1. Dieter, G.E., Bacon, D. (2013). _Mechanical Metallurgy_ , SI Metric ed. McGraw-Hill, pp. 189-245, 345-389.
  2. Courtney, T.H. (2005). _Mechanical Behavior of Materials_ , 2nd ed. Waveland Press, pp. 145-201.
  3. Hosford, W.F. (2010). _Mechanical Behavior of Materials_ , 2nd ed. Cambridge University Press, pp. 178-234.
  4. Porter, D.A., Easterling, K.E., Sherif, M.Y. (2009). _Phase Transformations in Metals and Alloys_ , 3rd ed. CRC Press, pp. 356-412.
  5. Smallman, R.E., Ngan, A.H.W. (2014). _Modern Physical Metallurgy_ , 8th ed. Butterworth-Heinemann, pp. 267-334.
  6. ASM Handbook Vol. 1 (1990). _Properties and Selection: Irons, Steels, and High-Performance Alloys_. ASM International, pp. 456-512.

[← Chapter 2](<./chapter-2.html>) [Chapter 4 →](<./chapter-4.html>)

### 免責事項

  * 本コンテンツは教育・研究・情報提供のみを目的としており、専門的な助言(法律・会計・技術的保証など)を提供するものではありません。
  * 本コンテンツおよび付随するCode examplesは「現状有姿(AS IS)」で提供され、明示または黙示を問わず、商品性、特定目的適合性、権利非侵害、正確性・完全性、動作・安全性等いかなる保証もしません。
  * 外部リンク、第三者が提供するデータ・ツール・ライブラリ等の内容・可用性・安全性について、作成者および東北大学は一切の責任を負いません。
  * 本コンテンツの利用・実行・解釈により直接的・間接的・付随的・特別・結果的・懲罰的損害が生じた場合でも、適用法で許容される最大限の範囲で、作成者および東北大学は責任を負いません。
  * 本コンテンツの内容は、予告なく変更・更新・提供停止されることがあります。
  * 本コンテンツの著作権・ライセンスは明記された条件(例: CC BY 4.0)に従います。当該ライセンスは通常、無保証条項を含みます。
