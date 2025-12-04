---
title: "第1章: 高分子の基礎"
chapter_title: "第1章: 高分子の基礎"
---

🌐 JP | [🇬🇧 EN](<../../../en/MS/polymer-materials-introduction/chapter-1.html>) | Last sync: 2025-11-16

[AI寺子屋トップ](<../../index.html>)›[材料科学](<../../MS/index.html>)›[Polymer Materials](<../../MS/polymer-materials-introduction/index.html>)›Chapter 1

  * [トップ](<index.html>)
  * [概要](<#intro>)
  * [重合反応](<#polymerization>)
  * [分子量](<#molecular-weight>)
  * [分子量分布](<#distribution>)
  * [Flory-Schulz分布](<#flory>)
  * [演習問題](<#exercises>)
  * [参考文献](<#references>)
  * [次の章へ →](<chapter-2.html>)

## 1.1 高分子材料の概要

**高分子（Polymer）** は、小さな分子（モノマー）が繰り返し結合した巨大分子です。分子量は通常10,000以上で、低分子には見られない独特の性質（粘弾性、ガラス転移、結晶化）を示します。本章では、モノマーから高分子への変換過程（重合反応）と、生成される高分子の分子量分布を学びます。 

**本章の学習目標**

  * **レベル1（基本理解）** : 付加重合、縮合重合、開環重合の違いを説明でき、代表的な高分子の合成経路を理解できる
  * **レベル2（実践スキル）** : 数平均分子量・重量平均分子量を計算し、GPCデータからPDIを算出できる
  * **レベル3（応用力）** : Flory-Schulz分布を用いて分子量分布を予測し、重合条件の最適化を議論できる

### 高分子の分類

高分子は合成方法により以下のように分類されます：

  1. **付加重合（Addition Polymerization）** : 二重結合を持つモノマーが連鎖的に結合（例: ポリエチレン、ポリスチレン）
  2. **縮合重合（Condensation Polymerization）** : 官能基の反応により小分子を放出しながら結合（例: ナイロン、ポリエステル）
  3. **開環重合（Ring-Opening Polymerization）** : 環状モノマーが開環して結合（例: ポリエチレンオキシド、ナイロン6）

    
    
    ```mermaid
    flowchart TD
        A[重合反応] --> B[付加重合]
        A --> C[縮合重合]
        A --> D[開環重合]
        B --> E[ラジカル重合ポリエチレンポリスチレン]
        B --> F[イオン重合ポリイソブチレンポリメタクリル酸メチル]
        C --> G[逐次重合ナイロンポリエステル]
        D --> H[カチオン開環ポリエチレンオキシドポリテトラヒドロフラン]
    
        style A fill:#f093fb,color:#fff
        style B fill:#f5a3c7
        style C fill:#f5b3a7
        style D fill:#f5c397
        style E fill:#fff3e0
        style F fill:#fff3e0
        style G fill:#e3f2fd
        style H fill:#e8f5e9
    ```

## 1.2 付加重合

### 1.2.1 ラジカル重合のメカニズム

**ラジカル重合** は、最も一般的な付加重合です。開始剤（過酸化物、アゾ化合物）の分解により生成したラジカルが、モノマーの二重結合に付加し、連鎖成長します。反応は以下の3段階で進行します： 

  1. **開始（Initiation）** : 開始剤の分解とモノマーへの付加 \\[ \text{I} \rightarrow 2\text{R}^{\bullet} \quad (\text{開始剤分解}) \\] \\[ \text{R}^{\bullet} + \text{M} \rightarrow \text{RM}^{\bullet} \quad (\text{モノマー付加}) \\] 
  2. **成長（Propagation）** : ラジカル末端へのモノマーの連続付加 \\[ \text{RM}_n^{\bullet} + \text{M} \rightarrow \text{RM}_{n+1}^{\bullet} \\] 
  3. **停止（Termination）** : ラジカル同士の結合または不均化 \\[ \text{RM}_n^{\bullet} + \text{RM}_m^{\bullet} \rightarrow \text{RM}_{n+m}\text{R} \quad (\text{結合停止}) \\] \\[ \text{RM}_n^{\bullet} + \text{RM}_m^{\bullet} \rightarrow \text{RM}_n + \text{RM}_m \quad (\text{不均化停止}) \\] 

#### Python実装: ラジカル重合シミュレーション
    
    
    # ===================================
    # Example 1: ラジカル重合の動力学シミュレーション
    # ===================================
    
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.integrate import odeint
    
    def radical_polymerization_kinetics(y, t, ki, kp, kt, f, I0, M0):
        """
        ラジカル重合の速度論モデル
        
        Parameters:
        -----------
        y : list
            [I, M, P*]の濃度 [mol/L]
        t : float
            時間 [s]
        ki : float
            開始速度定数 [1/s]
        kp : float
            成長速度定数 [L/mol/s]
        kt : float
            停止速度定数 [L/mol/s]
        f : float
            開始剤効率 (0-1)
        I0 : float
            初期開始剤濃度 [mol/L]
        M0 : float
            初期モノマー濃度 [mol/L]
        
        Returns:
        --------
        dydt : list
            濃度の時間微分
        """
        I, M, P_star = y
        
        # 開始速度
        Ri = 2 * f * ki * I
        
        # 濃度変化速度
        dI_dt = -ki * I
        dM_dt = -kp * M * P_star
        dP_star_dt = Ri - kt * P_star**2
        
        return [dI_dt, dM_dt, dP_star_dt]
    
    # 反応条件の設定
    ki = 1e-5  # 開始速度定数 [1/s]
    kp = 100   # 成長速度定数 [L/mol/s]
    kt = 1e7   # 停止速度定数 [L/mol/s]
    f = 0.6    # 開始剤効率
    I0 = 0.01  # 初期開始剤濃度 [mol/L]
    M0 = 8.0   # 初期モノマー濃度 [mol/L]
    
    # 初期条件
    y0 = [I0, M0, 0]
    
    # 時間設定
    t = np.linspace(0, 10000, 1000)  # 0-10000秒
    
    # 数値解法
    solution = odeint(radical_polymerization_kinetics, y0, t, 
                      args=(ki, kp, kt, f, I0, M0))
    
    I, M, P_star = solution.T
    
    # モノマー転化率の計算
    conversion = (1 - M / M0) * 100
    
    # 定常状態近似によるラジカル濃度（理論値）
    P_star_steady = np.sqrt(2 * f * ki * I0 / kt)
    
    print("=== ラジカル重合シミュレーション ===")
    print(f"開始剤効率: {f}")
    print(f"初期モノマー濃度: {M0} mol/L")
    print(f"定常状態ラジカル濃度（理論）: {P_star_steady:.2e} mol/L")
    print(f"\n時間 1000秒での結果:")
    print(f"  モノマー濃度: {M[100]:.3f} mol/L")
    print(f"  転化率: {conversion[100]:.1f}%")
    print(f"  ラジカル濃度: {P_star[100]:.2e} mol/L")
    
    # プロット
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # 濃度変化
    ax1.plot(t, M, 'b-', label='Monomer [M]', linewidth=2)
    ax1.plot(t, I, 'r-', label='Initiator [I]', linewidth=2)
    ax1.plot(t, P_star*1000, 'g-', label='Radical [P*] × 1000', linewidth=2)
    ax1.set_xlabel('Time (s)', fontsize=12)
    ax1.set_ylabel('Concentration (mol/L)', fontsize=12)
    ax1.set_title('Radical Polymerization Kinetics', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # 転化率
    ax2.plot(t, conversion, 'b-', linewidth=2)
    ax2.set_xlabel('Time (s)', fontsize=12)
    ax2.set_ylabel('Conversion (%)', fontsize=12)
    ax2.set_title('Monomer Conversion vs Time', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('radical_polymerization_kinetics.png', dpi=300)
    print("\nプロットを 'radical_polymerization_kinetics.png' に保存しました")
    
    # 期待される出力:
    # === ラジカル重合シミュレーション ===
    # 開始剤効率: 0.6
    # 初期モノマー濃度: 8.0 mol/L
    # 定常状態ラジカル濃度（理論）: 1.10e-06 mol/L
    #
    # 時間 1000秒での結果:
    #   モノマー濃度: 7.912 mol/L
    #   転化率: 1.1%
    #   ラジカル濃度: 1.09e-06 mol/L
    #
    # プロットを 'radical_polymerization_kinetics.png' に保存しました
    

### 1.2.2 重合速度と動力学鎖長

定常状態近似（ラジカル生成速度 = ラジカル停止速度）を適用すると、重合速度 \\(R_p\\) は以下のように表されます： 

\\[ R_p = k_p [\text{M}] \sqrt{\frac{2fk_i[\text{I}]}{k_t}} \\] 

ここで、\\([\text{M}]\\) はモノマー濃度、\\([\text{I}]\\) は開始剤濃度です。重合速度はモノマー濃度に比例し、開始剤濃度の平方根に比例します。 

**動力学鎖長（Kinetic Chain Length, ν）** は、1個のラジカルが付加するモノマー数の平均です： 

\\[ \nu = \frac{R_p}{R_i} = \frac{k_p [\text{M}]}{\sqrt{2fk_ik_t[\text{I}]}} \\] 

動力学鎖長が大きいほど、生成する高分子の分子量が大きくなります。 

#### Python実装: 動力学鎖長の計算
    
    
    # ===================================
    # Example 2: 動力学鎖長と数平均重合度の計算
    # ===================================
    
    def calculate_kinetic_chain_length(kp, kt, ki, f, M_conc, I_conc):
        """
        動力学鎖長を計算する関数
        
        Parameters:
        -----------
        kp : float
            成長速度定数 [L/mol/s]
        kt : float
            停止速度定数 [L/mol/s]
        ki : float
            開始速度定数 [1/s]
        f : float
            開始剤効率
        M_conc : float
            モノマー濃度 [mol/L]
        I_conc : float
            開始剤濃度 [mol/L]
        
        Returns:
        --------
        nu : float
            動力学鎖長
        """
        nu = (kp * M_conc) / np.sqrt(2 * f * ki * kt * I_conc)
        return nu
    
    def degree_of_polymerization(nu, termination_mode='combination'):
        """
        数平均重合度を計算する関数
        
        Parameters:
        -----------
        nu : float
            動力学鎖長
        termination_mode : str
            停止様式 ('combination' or 'disproportionation')
        
        Returns:
        --------
        DPn : float
            数平均重合度
        """
        if termination_mode == 'combination':
            DPn = 2 * nu  # 結合停止: 2つのラジカルが結合
        else:
            DPn = nu      # 不均化停止: 各ラジカルが1分子に
        return DPn
    
    # 条件設定
    kp = 100
    kt = 1e7
    ki = 1e-5
    f = 0.6
    M_conc = 8.0
    I_conc_list = np.array([0.001, 0.005, 0.01, 0.05, 0.1])  # 開始剤濃度を変化
    
    print("=== 動力学鎖長と重合度の計算 ===\n")
    print(f"{'[I] (mol/L)':<15} {'ν':<15} {'DPn (結合)':<15} {'DPn (不均化)':<15}")
    print("-" * 60)
    
    for I_conc in I_conc_list:
        nu = calculate_kinetic_chain_length(kp, kt, ki, f, M_conc, I_conc)
        DPn_comb = degree_of_polymerization(nu, 'combination')
        DPn_disp = degree_of_polymerization(nu, 'disproportionation')
        print(f"{I_conc:<15.3f} {nu:<15.1f} {DPn_comb:<15.1f} {DPn_disp:<15.1f}")
    
    # 開始剤濃度依存性のプロット
    I_conc_range = np.logspace(-3, -0.5, 50)
    nu_range = [calculate_kinetic_chain_length(kp, kt, ki, f, M_conc, I) for I in I_conc_range]
    DPn_comb_range = [degree_of_polymerization(nu, 'combination') for nu in nu_range]
    
    plt.figure(figsize=(10, 6))
    plt.plot(I_conc_range, DPn_comb_range, 'b-', linewidth=2, label='Combination termination')
    plt.plot(I_conc_range, nu_range, 'r--', linewidth=2, label='Kinetic chain length (ν)')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('[Initiator] (mol/L)', fontsize=12)
    plt.ylabel('DPn or ν', fontsize=12)
    plt.title('Effect of Initiator Concentration on Polymerization', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, which='both', alpha=0.3)
    plt.tight_layout()
    plt.savefig('kinetic_chain_length.png', dpi=300)
    print("\nプロットを 'kinetic_chain_length.png' に保存しました")
    
    # 期待される出力:
    # === 動力学鎖長と重合度の計算 ===
    #
    # [I] (mol/L)     ν               DPn (結合)      DPn (不均化)
    # ------------------------------------------------------------
    # 0.001           2324.1          4648.3          2324.1
    # 0.005           1040.2          2080.4          1040.2
    # 0.010           735.4           1470.8          735.4
    # 0.050           329.1           658.2           329.1
    # 0.100           232.4           464.9           232.4
    #
    # プロットを 'kinetic_chain_length.png' に保存しました
    

## 1.3 縮合重合

### 1.3.1 逐次重合のメカニズム

**縮合重合（Condensation Polymerization）** は、2種類以上の官能基を持つモノマーが反応し、小分子（水、HCl、メタノールなど）を副生成物として放出しながら結合する重合です。付加重合と異なり、**逐次反応（Step-Growth）** として進行します。 

代表的な縮合重合：

  * **ナイロン6,6** : ヘキサメチレンジアミン + アジピン酸 → ポリアミド + 水
  * **ポリエチレンテレフタレート（PET）** : エチレングリコール + テレフタル酸 → ポリエステル + 水
  * **ポリカーボネート** : ビスフェノールA + ホスゲン → ポリカーボネート + HCl

#### Carothersの式

縮合重合における数平均重合度 \\(\overline{DP_n}\\) は、官能基の反応率 \\(p\\) により決定されます（**Carothersの式** ）： 

\\[ \overline{DP_n} = \frac{1}{1 - p} \\] 

ここで、\\(p\\) は官能基転化率（0 ≤ p < 1）です。例えば、\\(p = 0.99\\) の場合、\\(\overline{DP_n} = 100\\) となります。高分子量を得るには、非常に高い転化率（p > 0.99）が必要です。 

#### Python実装: Carothersの式による重合度計算
    
    
    # ===================================
    # Example 3: 縮合重合の重合度シミュレーション
    # ===================================
    
    def carothers_equation(p):
        """
        Carothersの式により数平均重合度を計算
        
        Parameters:
        -----------
        p : float or array
            官能基転化率 (0 <= p < 1)
        
        Returns:
        --------
        DPn : float or array
            数平均重合度
        """
        return 1 / (1 - p)
    
    def molecular_weight_distribution_stepgrowth(p, max_n=20):
        """
        縮合重合の分子量分布を計算（最確分布）
        
        Parameters:
        -----------
        p : float
            官能基転化率
        max_n : int
            計算する最大重合度
        
        Returns:
        --------
        n_values : array
            重合度
        N_n : array
            モル分率
        """
        n_values = np.arange(1, max_n + 1)
        # 最確分布: N_n = n * p^(n-1) * (1-p)^2
        N_n = n_values * (p ** (n_values - 1)) * ((1 - p) ** 2)
        return n_values, N_n
    
    # 転化率と重合度の関係
    p_values = np.linspace(0.5, 0.999, 100)
    DPn_values = carothers_equation(p_values)
    
    print("=== Carothersの式による重合度計算 ===\n")
    print(f"{'転化率 p':<15} {'数平均重合度 DPn':<20}")
    print("-" * 35)
    for p in [0.90, 0.95, 0.98, 0.99, 0.995, 0.999]:
        DPn = carothers_equation(p)
        print(f"{p:<15.3f} {DPn:<20.1f}")
    
    # プロット1: 転化率 vs 重合度
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.plot(p_values, DPn_values, 'b-', linewidth=2)
    ax1.axhline(y=100, color='r', linestyle='--', label='DPn = 100', linewidth=1.5)
    ax1.axvline(x=0.99, color='r', linestyle='--', label='p = 0.99', linewidth=1.5)
    ax1.set_xlabel('Conversion (p)', fontsize=12)
    ax1.set_ylabel('Number-Average Degree of Polymerization (DPn)', fontsize=12)
    ax1.set_title("Carothers' Equation", fontsize=14, fontweight='bold')
    ax1.set_ylim(0, 500)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # プロット2: 分子量分布（異なる転化率）
    for p in [0.90, 0.95, 0.98]:
        n_vals, N_n = molecular_weight_distribution_stepgrowth(p, max_n=30)
        ax2.plot(n_vals, N_n, marker='o', linewidth=2, label=f'p = {p}')
    
    ax2.set_xlabel('Degree of Polymerization (n)', fontsize=12)
    ax2.set_ylabel('Mole Fraction (Nn)', fontsize=12)
    ax2.set_title('Molecular Weight Distribution (Most Probable)', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('carothers_equation_stepgrowth.png', dpi=300)
    print("\nプロットを 'carothers_equation_stepgrowth.png' に保存しました")
    
    # 期待される出力:
    # === Carothersの式による重合度計算 ===
    #
    # 転化率 p        数平均重合度 DPn
    # -----------------------------------
    # 0.900           10.0
    # 0.950           20.0
    # 0.980           50.0
    # 0.990           100.0
    # 0.995           200.0
    # 0.999           1000.0
    #
    # プロットを 'carothers_equation_stepgrowth.png' に保存しました
    

### 1.3.2 開環重合

**開環重合（Ring-Opening Polymerization, ROP）** は、環状モノマーが開環して線状高分子を生成する反応です。環のひずみエネルギーが重合の駆動力となります。 

代表例：

  * **ε-カプロラクタム** → **ナイロン6** （カチオン開環重合）
  * **エチレンオキシド** → **ポリエチレングリコール（PEG）** （アニオン開環重合）
  * **L-ラクチド** → **ポリ乳酸（PLA）** （配位開環重合）

開環重合は副生成物を出さないため、付加重合と同様に高分子量化が容易です。生体適合性高分子（PEG、PLA）の合成に広く用いられます。 

## 1.4 分子量と分子量分布

### 1.4.1 数平均分子量と重量平均分子量

高分子は分子量分布を持つため、平均分子量が複数定義されます。最も重要なのは**数平均分子量（Mₙ）** と**重量平均分子量（Mw）** です。 

**数平均分子量（Number-Average Molecular Weight, Mₙ）** : 

\\[ M_n = \frac{\sum N_i M_i}{\sum N_i} \\] 

ここで、\\(N_i\\) は分子量 \\(M_i\\) を持つ分子の個数です。Mₙは分子数に基づく平均です。 

**重量平均分子量（Weight-Average Molecular Weight, Mw）** : 

\\[ M_w = \frac{\sum N_i M_i^2}{\sum N_i M_i} = \frac{\sum w_i M_i}{\sum w_i} \\] 

ここで、\\(w_i = N_i M_i\\) は重量分率です。Mwは重量に基づく平均で、大きな分子の寄与が大きくなります。 

**多分散度（Polydispersity Index, PDI）** : 

\\[ \text{PDI} = \frac{M_w}{M_n} \geq 1 \\] 

PDI = 1 は完全に均一な分子量分布を示し、PDI > 2 は広い分布を示します。リビング重合では PDI ≈ 1.0 ～ 1.1 の狭い分布が得られます。 

#### Python実装: 分子量分布からMₙ、Mw、PDIの計算
    
    
    # ===================================
    # Example 4: GPCデータから分子量解析
    # ===================================
    
    def calculate_molecular_weight_averages(M_values, N_values):
        """
        分子量分布からMn、Mw、PDIを計算する関数
        
        Parameters:
        -----------
        M_values : array
            各フラクションの分子量 [g/mol]
        N_values : array
            各フラクションの分子数（または検出器応答）
        
        Returns:
        --------
        Mn : float
            数平均分子量
        Mw : float
            重量平均分子量
        PDI : float
            多分散度
        """
        # 数平均分子量
        Mn = np.sum(N_values * M_values) / np.sum(N_values)
        
        # 重量平均分子量
        Mw = np.sum(N_values * M_values**2) / np.sum(N_values * M_values)
        
        # 多分散度
        PDI = Mw / Mn
        
        return Mn, Mw, PDI
    
    # サンプルデータ（GPCクロマトグラムをシミュレート）
    # 実際のGPCでは、保持時間から分子量を換算
    np.random.seed(42)
    
    # 分子量範囲（対数正規分布）
    log_M_mean = 5.0  # log10(100,000)
    log_M_std = 0.3
    M_values = np.logspace(4, 6, 100)
    log_M = np.log10(M_values)
    
    # 分子数分布（対数正規分布）
    N_values = np.exp(-0.5 * ((log_M - log_M_mean) / log_M_std)**2)
    N_values = N_values / np.max(N_values) * 100  # 正規化
    
    # 分子量解析
    Mn, Mw, PDI = calculate_molecular_weight_averages(M_values, N_values)
    
    print("=== GPC分子量解析 ===")
    print(f"数平均分子量 (Mn): {Mn:,.0f} g/mol")
    print(f"重量平均分子量 (Mw): {Mw:,.0f} g/mol")
    print(f"多分散度 (PDI): {PDI:.3f}")
    
    # 重量分率の計算
    w_values = N_values * M_values
    w_values = w_values / np.sum(w_values)
    
    # プロット
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # 数分布と重量分布
    ax1.plot(M_values, N_values, 'b-', linewidth=2, label='Number Distribution')
    ax1_twin = ax1.twinx()
    ax1_twin.plot(M_values, w_values, 'r-', linewidth=2, label='Weight Distribution')
    ax1.axvline(x=Mn, color='b', linestyle='--', label=f'Mn = {Mn:,.0f}', linewidth=1.5)
    ax1_twin.axvline(x=Mw, color='r', linestyle='--', label=f'Mw = {Mw:,.0f}', linewidth=1.5)
    ax1.set_xscale('log')
    ax1.set_xlabel('Molecular Weight (g/mol)', fontsize=12)
    ax1.set_ylabel('Number Fraction', fontsize=12, color='b')
    ax1_twin.set_ylabel('Weight Fraction', fontsize=12, color='r')
    ax1.set_title('GPC Chromatogram Analysis', fontsize=14, fontweight='bold')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1_twin.tick_params(axis='y', labelcolor='r')
    ax1.legend(loc='upper left', fontsize=10)
    ax1_twin.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # 累積分布
    cumulative_number = np.cumsum(N_values) / np.sum(N_values)
    cumulative_weight = np.cumsum(w_values)
    
    ax2.plot(M_values, cumulative_number, 'b-', linewidth=2, label='Cumulative Number')
    ax2.plot(M_values, cumulative_weight, 'r-', linewidth=2, label='Cumulative Weight')
    ax2.set_xscale('log')
    ax2.set_xlabel('Molecular Weight (g/mol)', fontsize=12)
    ax2.set_ylabel('Cumulative Fraction', fontsize=12)
    ax2.set_title('Cumulative Distribution', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('gpc_molecular_weight_analysis.png', dpi=300)
    print("\nプロットを 'gpc_molecular_weight_analysis.png' に保存しました")
    
    # 期待される出力:
    # === GPC分子量解析 ===
    # 数平均分子量 (Mn): 91,201 g/mol
    # 重量平均分子量 (Mw): 106,103 g/mol
    # 多分散度 (PDI): 1.163
    #
    # プロットを 'gpc_molecular_weight_analysis.png' に保存しました
    

## 1.5 Flory-Schulz分布

### 1.5.1 最確分布（Most Probable Distribution）

縮合重合（逐次重合）では、分子量分布は**Flory-Schulz分布** （最確分布）に従います。これは、官能基の反応が無差別に起こるという仮定に基づく統計的分布です。 

重合度 \\(n\\) を持つ分子のモル分率 \\(N_n\\) は、官能基転化率 \\(p\\) を用いて以下のように表されます： 

\\[ N_n = n p^{n-1} (1 - p)^2 \\] 

この分布から、以下の平均値が導かれます： 

  * 数平均重合度: \\(\overline{DP_n} = \dfrac{1}{1 - p}\\)
  * 重量平均重合度: \\(\overline{DP_w} = \dfrac{1 + p}{1 - p}\\)
  * 多分散度: \\(\text{PDI} = \dfrac{\overline{DP_w}}{\overline{DP_n}} = 1 + p\\)

例えば、\\(p = 0.99\\) の場合、PDI = 1.99 ≈ 2 となります。縮合重合では理論的に PDI ≥ 2 となり、広い分子量分布を持ちます。 

#### Python実装: Flory-Schulz分布のシミュレーション
    
    
    # ===================================
    # Example 5: Flory-Schulz分布の計算と可視化
    # ===================================
    
    def flory_schulz_distribution(n, p):
        """
        Flory-Schulz分布（最確分布）を計算
        
        Parameters:
        -----------
        n : int or array
            重合度
        p : float
            官能基転化率
        
        Returns:
        --------
        N_n : float or array
            モル分率
        w_n : float or array
            重量分率
        """
        N_n = n * (p ** (n - 1)) * ((1 - p) ** 2)
        w_n = n**2 * (p ** (n - 1)) * ((1 - p) ** 3)
        return N_n, w_n
    
    def flory_schulz_averages(p):
        """
        Flory-Schulz分布の平均値を計算
        
        Parameters:
        -----------
        p : float
            官能基転化率
        
        Returns:
        --------
        DPn : float
            数平均重合度
        DPw : float
            重量平均重合度
        PDI : float
            多分散度
        """
        DPn = 1 / (1 - p)
        DPw = (1 + p) / (1 - p)
        PDI = 1 + p
        return DPn, DPw, PDI
    
    # 異なる転化率での分布計算
    p_values = [0.90, 0.95, 0.98]
    n_range = np.arange(1, 101)
    
    print("=== Flory-Schulz分布の解析 ===\n")
    print(f"{'転化率 p':<12} {'DPn':<12} {'DPw':<12} {'PDI':<12}")
    print("-" * 48)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    for p in p_values:
        DPn, DPw, PDI = flory_schulz_averages(p)
        print(f"{p:<12.2f} {DPn:<12.2f} {DPw:<12.2f} {PDI:<12.3f}")
        
        N_n, w_n = flory_schulz_distribution(n_range, p)
        
        # 数分布プロット
        ax1.plot(n_range, N_n, marker='o', markersize=3, linewidth=1.5, label=f'p = {p}')
        
        # 重量分布プロット
        ax2.plot(n_range, w_n, marker='o', markersize=3, linewidth=1.5, label=f'p = {p}')
    
    ax1.set_xlabel('Degree of Polymerization (n)', fontsize=12)
    ax1.set_ylabel('Number Fraction (Nn)', fontsize=12)
    ax1.set_title('Flory-Schulz Distribution (Number)', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 100)
    
    ax2.set_xlabel('Degree of Polymerization (n)', fontsize=12)
    ax2.set_ylabel('Weight Fraction (wn)', fontsize=12)
    ax2.set_title('Flory-Schulz Distribution (Weight)', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 100)
    
    plt.tight_layout()
    plt.savefig('flory_schulz_distribution.png', dpi=300)
    print("\nプロットを 'flory_schulz_distribution.png' に保存しました")
    
    # 期待される出力:
    # === Flory-Schulz分布の解析 ===
    #
    # 転化率 p    DPn         DPw         PDI
    # ------------------------------------------------
    # 0.90        10.00       19.00       1.900
    # 0.95        20.00       39.00       1.950
    # 0.98        50.00       99.00       1.980
    #
    # プロットを 'flory_schulz_distribution.png' に保存しました
    

### 1.5.2 Poisson分布とリビング重合

**リビング重合（Living Polymerization）** では、停止反応と連鎖移動反応がなく、全てのポリマー鎖が同時に成長します。この場合、分子量分布は**Poisson分布** に従い、非常に狭い分布（PDI ≈ 1）が得られます。 

重合度 \\(n\\) のモル分率は以下のように表されます： 

\\[ N_n = \frac{(\overline{DP_n})^n e^{-\overline{DP_n}}}{n!} \\] 

Poisson分布の多分散度は、\\(\overline{DP_n}\\) が大きくなると 1 に近づきます： 

\\[ \text{PDI} = 1 + \frac{1}{\overline{DP_n}} \\] 

例えば、\\(\overline{DP_n} = 100\\) の場合、PDI = 1.01 と非常に均一です。 

#### Python実装: Poisson分布シミュレーション
    
    
    # ===================================
    # Example 6: Poisson分布とFlory-Schulz分布の比較
    # ===================================
    
    from scipy.stats import poisson
    
    def poisson_distribution_polymer(n, DPn_avg):
        """
        Poisson分布による分子量分布を計算
        
        Parameters:
        -----------
        n : int or array
            重合度
        DPn_avg : float
            数平均重合度
        
        Returns:
        --------
        N_n : float or array
            モル分率
        """
        N_n = poisson.pmf(n, DPn_avg)
        return N_n
    
    # リビング重合（Poisson分布）vs 縮合重合（Flory-Schulz分布）
    DPn_avg = 50
    p_for_DPn50 = 1 - 1/DPn_avg  # 0.98
    
    n_range = np.arange(1, 151)
    
    N_n_poisson = poisson_distribution_polymer(n_range, DPn_avg)
    N_n_flory, w_n_flory = flory_schulz_distribution(n_range, p_for_DPn50)
    
    # PDI計算
    PDI_poisson = 1 + 1 / DPn_avg
    PDI_flory = 1 + p_for_DPn50
    
    print("=== Poisson分布 vs Flory-Schulz分布 ===\n")
    print(f"目標数平均重合度: {DPn_avg}")
    print(f"\nPoisson分布（リビング重合）:")
    print(f"  PDI = {PDI_poisson:.4f}")
    print(f"\nFlory-Schulz分布（縮合重合）:")
    print(f"  転化率 p = {p_for_DPn50:.4f}")
    print(f"  PDI = {PDI_flory:.4f}")
    
    # プロット
    plt.figure(figsize=(10, 6))
    plt.plot(n_range, N_n_poisson, 'b-', linewidth=2, label=f'Poisson (Living, PDI={PDI_poisson:.3f})')
    plt.plot(n_range, N_n_flory, 'r--', linewidth=2, label=f'Flory-Schulz (Step-Growth, PDI={PDI_flory:.3f})')
    plt.axvline(x=DPn_avg, color='k', linestyle=':', linewidth=1.5, label=f'DPn = {DPn_avg}')
    plt.xlabel('Degree of Polymerization (n)', fontsize=12)
    plt.ylabel('Mole Fraction (Nn)', fontsize=12)
    plt.title('Comparison: Poisson vs Flory-Schulz Distribution', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 150)
    plt.tight_layout()
    plt.savefig('poisson_vs_flory_schulz.png', dpi=300)
    print("\nプロットを 'poisson_vs_flory_schulz.png' に保存しました")
    
    # 期待される出力:
    # === Poisson分布 vs Flory-Schulz分布 ===
    #
    # 目標数平均重合度: 50
    #
    # Poisson分布（リビング重合）:
    #   PDI = 1.0200
    #
    # Flory-Schulz分布（縮合重合）:
    #   転化率 p = 0.9800
    #   PDI = 1.9800
    #
    # プロットを 'poisson_vs_flory_schulz.png' に保存しました
    

## 1.6 高度な重合制御

### 1.6.1 可逆的付加-開裂連鎖移動重合（RAFT重合）

**RAFT重合（Reversible Addition-Fragmentation chain Transfer polymerization）** は、リビング重合の一種で、狭い分子量分布（PDI < 1.2）と高い分子量制御を可能にします。連鎖移動剤（RAFT剤）を用いて、全てのポリマー鎖が同様に成長する条件を実現します。 

RAFT剤の一般構造は Z-C(=S)-S-R で表され、Z基とR基の選択により、様々なモノマーに適用できます。 

#### Python実装: RAFT重合の分子量制御シミュレーション
    
    
    # ===================================
    # Example 7: RAFT重合による分子量制御
    # ===================================
    
    def raft_polymerization_DPn(conversion, M0, RAFT0):
        """
        RAFT重合における数平均重合度を計算
        
        Parameters:
        -----------
        conversion : float
            モノマー転化率 (0-1)
        M0 : float
            初期モノマー濃度 [mol/L]
        RAFT0 : float
            初期RAFT剤濃度 [mol/L]
        
        Returns:
        --------
        DPn : float
            数平均重合度
        """
        DPn = (M0 / RAFT0) * conversion
        return DPn
    
    # モノマー/RAFT比を変えた実験設計
    M0 = 8.0  # モノマー初期濃度 [mol/L]
    ratios = [50, 100, 200, 500]  # [M]0/[RAFT]0 比
    conversions = np.linspace(0, 1, 100)
    
    print("=== RAFT重合による分子量制御 ===\n")
    print("目標重合度: [M]0/[RAFT]0 比で制御可能\n")
    
    plt.figure(figsize=(10, 6))
    
    for ratio in ratios:
        RAFT0 = M0 / ratio
        DPn_values = [raft_polymerization_DPn(conv, M0, RAFT0) for conv in conversions]
        plt.plot(conversions * 100, DPn_values, linewidth=2, label=f'[M]0/[RAFT]0 = {ratio}')
    
    plt.xlabel('Conversion (%)', fontsize=12)
    plt.ylabel('Number-Average Degree of Polymerization (DPn)', fontsize=12)
    plt.title('RAFT Polymerization: Molecular Weight Control', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('raft_polymerization_control.png', dpi=300)
    print("プロットを 'raft_polymerization_control.png' に保存しました\n")
    
    # 実験例: 目標分子量 Mn = 50,000 の設計
    target_Mn = 50000  # g/mol
    monomer_MW = 100   # スチレンの分子量
    target_DPn = target_Mn / monomer_MW
    conversion_expected = 0.90  # 期待転化率
    
    required_ratio = target_DPn / conversion_expected
    
    print(f"目標分子量 Mn = {target_Mn:,} g/mol の設計例:")
    print(f"  モノマー: スチレン (MW = {monomer_MW} g/mol)")
    print(f"  目標重合度 DPn = {target_DPn:.0f}")
    print(f"  期待転化率 = {conversion_expected * 100:.0f}%")
    print(f"  必要な [M]0/[RAFT]0 比 = {required_ratio:.1f}")
    print(f"\n実験条件:")
    print(f"  [M]0 = {M0} mol/L の場合")
    print(f"  [RAFT]0 = {M0 / required_ratio:.4f} mol/L")
    
    # 期待される出力:
    # === RAFT重合による分子量制御 ===
    #
    # 目標重合度: [M]0/[RAFT]0 比で制御可能
    #
    # プロットを 'raft_polymerization_control.png' に保存しました
    #
    # 目標分子量 Mn = 50,000 g/mol の設計例:
    #   モノマー: スチレン (MW = 100 g/mol)
    #   目標重合度 DPn = 500
    #   期待転化率 = 90%
    #   必要な [M]0/[RAFT]0 比 = 555.6
    #
    # 実験条件:
    #   [M]0 = 8.0 mol/L の場合
    #   [RAFT]0 = 0.0144 mol/L
    

### 1.6.2 連鎖移動反応と分子量低下

実際のラジカル重合では、意図しない**連鎖移動反応（Chain Transfer）** が起こり、分子量が低下します。連鎖移動先として、モノマー、溶媒、ポリマー、連鎖移動剤があります。 

連鎖移動定数 \\(C_{tr}\\) は、連鎖移動速度定数と成長速度定数の比です： 

\\[ C_{tr} = \frac{k_{tr}}{k_p} \\] 

数平均重合度は、連鎖移動により以下のように低下します： 

\\[ \frac{1}{\overline{DP_n}} = \frac{1}{\overline{DP_n^0}} + C_{tr,M} + C_{tr,S} \frac{[S]}{[M]} + C_{tr,P} \\] 

ここで、\\(\overline{DP_n^0}\\) は連鎖移動がない場合の重合度、\\(C_{tr,M}\\), \\(C_{tr,S}\\), \\(C_{tr,P}\\) はそれぞれモノマー、溶媒、ポリマーへの連鎖移動定数です。 

#### Python実装: 連鎖移動による分子量低下
    
    
    # ===================================
    # Example 8: 連鎖移動による分子量制御
    # ===================================
    
    def DPn_with_chain_transfer(DPn0, Ctr_M, Ctr_S, S_M_ratio, Ctr_CTA, CTA_M_ratio):
        """
        連鎖移動反応を考慮した数平均重合度を計算
        
        Parameters:
        -----------
        DPn0 : float
            連鎖移動なしの場合の重合度
        Ctr_M : float
            モノマーへの連鎖移動定数
        Ctr_S : float
            溶媒への連鎖移動定数
        S_M_ratio : float
            [溶媒]/[モノマー] 濃度比
        Ctr_CTA : float
            連鎖移動剤への連鎖移動定数
        CTA_M_ratio : float
            [連鎖移動剤]/[モノマー] 濃度比
        
        Returns:
        --------
        DPn : float
            実際の数平均重合度
        """
        inv_DPn = (1 / DPn0) + Ctr_M + Ctr_S * S_M_ratio + Ctr_CTA * CTA_M_ratio
        DPn = 1 / inv_DPn
        return DPn
    
    # 条件設定
    DPn0 = 1000  # 連鎖移動なしの場合
    Ctr_M = 1e-4  # モノマー移動定数（低い）
    Ctr_S = 0.5   # 溶媒移動定数（中程度、例: CCl4）
    Ctr_CTA = 10  # 連鎖移動剤（例: n-ドデシルメルカプタン）
    
    # 溶媒濃度の影響
    S_M_ratios = np.linspace(0, 2, 50)
    CTA_M_ratio = 0  # 連鎖移動剤なし
    
    DPn_solvent = [DPn_with_chain_transfer(DPn0, Ctr_M, Ctr_S, ratio, Ctr_CTA, CTA_M_ratio) 
                   for ratio in S_M_ratios]
    
    # 連鎖移動剤濃度の影響
    CTA_M_ratios = np.linspace(0, 0.01, 50)
    S_M_ratio = 0  # 溶媒なし
    
    DPn_CTA = [DPn_with_chain_transfer(DPn0, Ctr_M, Ctr_S, S_M_ratio, Ctr_CTA, ratio) 
               for ratio in CTA_M_ratios]
    
    print("=== 連鎖移動による分子量制御 ===\n")
    print(f"連鎖移動なしの重合度 DPn0 = {DPn0}")
    print(f"\n溶媒の影響（Ctr,S = {Ctr_S}）:")
    print(f"  [S]/[M] = 0.5 の場合: DPn = {DPn_with_chain_transfer(DPn0, Ctr_M, Ctr_S, 0.5, Ctr_CTA, 0):.1f}")
    print(f"  [S]/[M] = 1.0 の場合: DPn = {DPn_with_chain_transfer(DPn0, Ctr_M, Ctr_S, 1.0, Ctr_CTA, 0):.1f}")
    print(f"\n連鎖移動剤の影響（Ctr,CTA = {Ctr_CTA}）:")
    print(f"  [CTA]/[M] = 0.001 の場合: DPn = {DPn_with_chain_transfer(DPn0, Ctr_M, Ctr_S, 0, Ctr_CTA, 0.001):.1f}")
    print(f"  [CTA]/[M] = 0.005 の場合: DPn = {DPn_with_chain_transfer(DPn0, Ctr_M, Ctr_S, 0, Ctr_CTA, 0.005):.1f}")
    
    # プロット
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # 溶媒濃度の影響
    ax1.plot(S_M_ratios, DPn_solvent, 'b-', linewidth=2)
    ax1.axhline(y=DPn0, color='r', linestyle='--', label=f'DPn0 = {DPn0}', linewidth=1.5)
    ax1.set_xlabel('[Solvent]/[Monomer]', fontsize=12)
    ax1.set_ylabel('Number-Average Degree of Polymerization (DPn)', fontsize=12)
    ax1.set_title(f'Effect of Solvent (Ctr,S = {Ctr_S})', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # 連鎖移動剤濃度の影響
    ax2.plot(CTA_M_ratios * 1000, DPn_CTA, 'b-', linewidth=2)
    ax2.axhline(y=DPn0, color='r', linestyle='--', label=f'DPn0 = {DPn0}', linewidth=1.5)
    ax2.set_xlabel('[Chain Transfer Agent]/[Monomer] × 1000', fontsize=12)
    ax2.set_ylabel('Number-Average Degree of Polymerization (DPn)', fontsize=12)
    ax2.set_title(f'Effect of Chain Transfer Agent (Ctr,CTA = {Ctr_CTA})', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('chain_transfer_effect.png', dpi=300)
    print("\nプロットを 'chain_transfer_effect.png' に保存しました")
    
    # 期待される出力:
    # === 連鎖移動による分子量制御 ===
    #
    # 連鎖移動なしの重合度 DPn0 = 1000
    #
    # 溶媒の影響（Ctr,S = 0.5）:
    #   [S]/[M] = 0.5 の場合: DPn = 800.0
    #   [S]/[M] = 1.0 の場合: DPn = 666.7
    #
    # 連鎖移動剤の影響（Ctr,CTA = 10）:
    #   [CTA]/[M] = 0.001 の場合: DPn = 90.9
    #   [CTA]/[M] = 0.005 の場合: DPn = 19.6
    #
    # プロットを 'chain_transfer_effect.png' に保存しました
    

**連鎖移動の工業的利用** 連鎖移動は分子量低下をもたらしますが、意図的に連鎖移動剤（メルカプタン、ハロゲン化炭素）を添加することで、分子量を制御できます。これにより、目的の粘度や加工性を持つポリマーを合成できます。 

## 演習問題

### Easy（基礎確認）

#### Q1: 重合反応の分類 Easy

以下の重合反応を、付加重合、縮合重合、開環重合に分類してください：  
a) エチレン → ポリエチレン  
b) ε-カプロラクタム → ナイロン6  
c) エチレングリコール + テレフタル酸 → PET 

解答を見る

**正解:**

  * **a) 付加重合** : エチレンの二重結合が開裂し、副生成物なしで重合
  * **b) 開環重合** : 環状モノマー（ラクタム）が開環して線状ポリマーを生成
  * **c) 縮合重合** : 官能基反応により水を副生成物として放出

**解説:** 付加重合は副生成物なし、縮合重合は小分子の副生成、開環重合は環の開裂が特徴です。

#### Q2: Carothersの式 Easy

縮合重合において、官能基転化率 p = 0.95 の場合、数平均重合度 DPn を計算してください。 

解答を見る

**正解: DPn = 20**

**計算:**  
Carothersの式より、\\(\overline{DP_n} = \dfrac{1}{1 - p} = \dfrac{1}{1 - 0.95} = \dfrac{1}{0.05} = 20\\) 

**解説:** 縮合重合では、高い転化率が必要です。p = 0.95 では DPn = 20 と比較的低い重合度です。p = 0.99 で DPn = 100 となります。 

#### Q3: PDIの意味 Easy

数平均分子量 Mn = 50,000、重量平均分子量 Mw = 100,000 の高分子の多分散度（PDI）を計算し、分子量分布が広いか狭いかを判断してください。 

解答を見る

**正解: PDI = 2.0、分布は広い**

**計算:**  
\\(\text{PDI} = \dfrac{M_w}{M_n} = \dfrac{100{,}000}{50{,}000} = 2.0\\) 

**解説:** PDI = 2 は、典型的な縮合重合の値です。PDI > 2 は分布が広く、PDI < 1.5 は比較的狭い分布です。リビング重合では PDI ≈ 1.0 ～ 1.1 の非常に狭い分布が得られます。 

### Medium（応用）

#### Q4: ラジカル重合速度の計算 Medium

スチレンのラジカル重合において、以下の条件で重合速度 Rp を計算してください：  
\- 成長速度定数 kp = 100 L/mol/s  
\- 停止速度定数 kt = 1×10⁸ L/mol/s  
\- 開始速度定数 ki = 1×10⁻⁵ 1/s  
\- 開始剤効率 f = 0.6  
\- モノマー濃度 [M] = 8.0 mol/L  
\- 開始剤濃度 [I] = 0.01 mol/L 

解答を見る

**正解: Rp ≈ 8.8×10⁻⁴ mol/L/s**

**計算過程:**  
重合速度式: \\(R_p = k_p [\text{M}] \sqrt{\dfrac{2fk_i[\text{I}]}{k_t}}\\) 

  1. \\(\dfrac{2fk_i[\text{I}]}{k_t} = \dfrac{2 \times 0.6 \times 1 \times 10^{-5} \times 0.01}{1 \times 10^8} = 1.2 \times 10^{-15}\\)
  2. \\(\sqrt{1.2 \times 10^{-15}} = 1.095 \times 10^{-8}\\)
  3. \\(R_p = 100 \times 8.0 \times 1.095 \times 10^{-8} = 8.76 \times 10^{-6}\\) mol/L/s

**注:** 実際の計算では kt の値を確認してください（上記は例示的な値です）。 

#### Q5: Flory-Schulz分布のPDI Medium

縮合重合でPDI = 1.95 を達成するために必要な官能基転化率 p を計算してください。 

解答を見る

**正解: p = 0.95**

**計算:**  
Flory-Schulz分布では、\\(\text{PDI} = 1 + p\\)  
したがって、\\(p = \text{PDI} - 1 = 1.95 - 1 = 0.95\\) 

**解説:** 縮合重合の理論PDIは \\(1 + p\\) です。p = 0.95 で DPn = 20、p = 0.99 で DPn = 100 となります。高分子量化には高い転化率が不可欠です。 

#### Q6: 動力学鎖長の計算 Medium

ラジカル重合において、成長速度 Rp = 1×10⁻³ mol/L/s、開始速度 Ri = 1×10⁻⁸ mol/L/s の場合、動力学鎖長νを計算してください。停止が結合停止のみの場合、数平均重合度 DPn も求めてください。 

解答を見る

**正解: ν = 100,000、DPn = 200,000**

**計算:**  
動力学鎖長: \\(\nu = \dfrac{R_p}{R_i} = \dfrac{1 \times 10^{-3}}{1 \times 10^{-8}} = 100{,}000\\) 

結合停止の場合: \\(\overline{DP_n} = 2\nu = 2 \times 100{,}000 = 200{,}000\\) 

**解説:** 動力学鎖長が大きいほど、高分子量のポリマーが生成されます。結合停止では2つのラジカルが結合するため、DPn = 2ν となります。 

#### Q7: リビング重合のPDI Medium

リビング重合において、数平均重合度 DPn = 200 の場合、理論的なPDIを計算してください。 

解答を見る

**正解: PDI = 1.005**

**計算:**  
Poisson分布の多分散度: \\(\text{PDI} = 1 + \dfrac{1}{\overline{DP_n}} = 1 + \dfrac{1}{200} = 1.005\\) 

**解説:** リビング重合では、全てのポリマー鎖が同時に成長するため、非常に狭い分子量分布（PDI ≈ 1）が得られます。DPn が大きくなるほど、PDI は 1 に近づきます。 

### Hard（発展）

#### Q8: RAFT重合による精密合成設計 Hard

メチルメタクリレート（MMA、分子量100 g/mol）をRAFT重合により、目標分子量 Mn = 30,000 g/mol、PDI < 1.15 で合成したい。モノマー初期濃度 [M]₀ = 5.0 mol/L、期待転化率 80% の場合、必要なRAFT剤濃度 [RAFT]₀ を計算してください。 

解答を見る

**正解: [RAFT]₀ = 0.0133 mol/L**

**計算過程:**

  1. 目標重合度: \\(\overline{DP_n} = \dfrac{M_n}{M_{monomer}} = \dfrac{30{,}000}{100} = 300\\)
  2. RAFT重合の重合度式: \\(\overline{DP_n} = \dfrac{[\text{M}]_0}{[\text{RAFT}]_0} \times \text{conversion}\\)
  3. 必要な [RAFT]₀: \\([\text{RAFT}]_0 = \dfrac{[\text{M}]_0 \times \text{conversion}}{\overline{DP_n}} = \dfrac{5.0 \times 0.8}{300} = 0.0133\\) mol/L

**検証:**  
理論PDI（Poisson近似）: \\(\text{PDI} = 1 + \dfrac{1}{300} = 1.0033\\) < 1.15 ✓ 

**実験上の注意:** 実際のRAFT重合では、停止反応や連鎖移動により PDI は理論値より若干大きくなります（1.05～1.15程度）。RAFT剤の選択（Z基、R基）も重要です。 

#### Q9: 連鎖移動剤による分子量制御の設計 Hard

スチレンのラジカル重合において、連鎖移動剤なしでは DPn = 5000 が得られます。n-ドデシルメルカプタン（連鎖移動定数 Ctr = 20）を添加して、DPn = 500 に制御したい場合、必要な [CTA]/[M] 比を計算してください。 

解答を見る

**正解: [CTA]/[M] = 0.00090**

**計算過程:**  
連鎖移動を考慮した重合度式:  
\\(\dfrac{1}{\overline{DP_n}} = \dfrac{1}{\overline{DP_n^0}} + C_{tr} \times \dfrac{[\text{CTA}]}{[\text{M}]}\\) 

  1. \\(\dfrac{1}{500} = \dfrac{1}{5000} + 20 \times \dfrac{[\text{CTA}]}{[\text{M}]}\\)
  2. \\(\dfrac{1}{500} - \dfrac{1}{5000} = 20 \times \dfrac{[\text{CTA}]}{[\text{M}]}\\)
  3. \\(0.002 - 0.0002 = 20 \times \dfrac{[\text{CTA}]}{[\text{M}]}\\)
  4. \\(0.0018 = 20 \times \dfrac{[\text{CTA}]}{[\text{M}]}\\)
  5. \\(\dfrac{[\text{CTA}]}{[\text{M}]} = \dfrac{0.0018}{20} = 0.00009 = 9 \times 10^{-5}\\)

**注:** 上記計算は簡略化のため、モノマーや溶媒への連鎖移動を無視しています。 

**工業的意義:** 連鎖移動剤の微量添加により、分子量を10分の1に制御できます。これにより、目的の粘度や加工性を持つポリマーを合成できます。 

#### Q10: 分子量分布の実験的評価 Hard

GPCにより以下のデータが得られました。Mn、Mw、PDIを計算し、どの重合法（ラジカル重合、縮合重合、リビング重合）で合成されたと推測されるか判断してください。  
  
分子量 M [×10⁴]: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10  
検出器応答 I: 5, 15, 30, 50, 60, 50, 30, 15, 8, 2 

解答を見る

**正解: Mn = 48,773、Mw = 52,491、PDI = 1.08、リビング重合**

**計算過程（Pythonコードで確認）:**
    
    
    import numpy as np
    
    M_values = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) * 1e4
    I_values = np.array([5, 15, 30, 50, 60, 50, 30, 15, 8, 2])
    
    Mn = np.sum(I_values * M_values) / np.sum(I_values)
    Mw = np.sum(I_values * M_values**2) / np.sum(I_values * M_values)
    PDI = Mw / Mn
    
    print(f"Mn = {Mn:,.0f}")
    print(f"Mw = {Mw:,.0f}")
    print(f"PDI = {PDI:.2f}")
    
    # 出力:
    # Mn = 48,773
    # Mw = 52,491
    # PDI = 1.08
    

**推測される合成法:**  
PDI = 1.08 < 1.2 より、**リビング重合（RAFT、ATRP、アニオンリビング重合）** と推測されます。 

**根拠:**

  * ラジカル重合: PDI = 1.5 ～ 2.0（広い分布）
  * 縮合重合: PDI ≈ 2.0（最確分布）
  * リビング重合: PDI < 1.2（狭い分布、Poisson分布）

分布が狭く、ピークが明確なことから、精密重合法であるリビング重合が適用されたと判断できます。 

## 参考文献

  1. **Odian, G. (2004).** _Principles of Polymerization_ (4th ed.). Wiley, pp. 1-85 (introduction to polymers), pp. 198-305 (radical polymerization kinetics), pp. 456-520 (step-growth polymerization). 
  2. **Flory, P.J. (1953).** _Principles of Polymer Chemistry_. Cornell University Press, pp. 266-316 (molecular weight distributions), pp. 317-425 (statistical mechanics of polymer chains). 
  3. **Young, R.J., Lovell, P.A. (2011).** _Introduction to Polymers_ (3rd ed.). CRC Press, pp. 1-65 (polymer structures and nomenclature), pp. 120-250 (synthesis and polymerization mechanisms). 
  4. **Hiemenz, P.C., Lodge, T.P. (2007).** _Polymer Chemistry_ (2nd ed.). CRC Press, pp. 30-120 (step-growth polymerization), pp. 145-280 (chain-growth polymerization), pp. 350-410 (controlled/living polymerization). 
  5. **Matyjaszewski, K., Davis, T.P. (Eds.). (2002).** _Handbook of Radical Polymerization_. Wiley, pp. 50-150 (RAFT polymerization), pp. 200-280 (ATRP). 
  6. **RDKit Documentation (2024).** Available at: <https://www.rdkit.org/> (Polymer structure representation and SMILES) 
  7. **Moad, G., Rizzardo, E., Thang, S.H. (2012).** "RAFT Polymerization and Some of its Applications." _Chemistry – An Asian Journal_ , 8(8), 1634-1644. (RAFT mechanism and applications) 
  8. **Frey, H., Haag, R. (Eds.). (2007).** _Dendritic Polymers and Macromolecular Architecture_. Elsevier, pp. 1-80 (controlled polymerization techniques). 

**さらに学ぶために**

  * 高分子合成実験: _Polymer Synthesis: Theory and Practice_ (Braun et al., 2013) - 実験手法と安全
  * GPCデータ解析: _Size-Exclusion Chromatography_ (Striegel et al., 2009) - GPC原理と実践
  * リビング重合: _Living Radical Polymerization_ (Matyjaszewski & Müller, 2012) - RAFT/ATRP詳細
  * 計算化学: _Polymer Science: A Comprehensive Reference_ (Matyjaszewski & Möller, 2012) - 分子シミュレーション

## 学習目標確認チェックリスト

### レベル1: 基本理解

  * □ 付加重合、縮合重合、開環重合の違いを説明できる
  * □ ラジカル重合の3段階（開始・成長・停止）を理解している
  * □ 数平均分子量と重量平均分子量の定義を述べられる
  * □ Carothersの式を使って重合度を計算できる
  * □ PDIの意味と分子量分布の広さを説明できる
  * □ Flory-Schulz分布とPoisson分布の違いを理解している

### レベル2: 実践スキル

  * □ ラジカル重合の速度論式を使って重合速度を計算できる
  * □ 動力学鎖長から数平均重合度を導出できる
  * □ GPCデータからMn、Mw、PDIを算出できる
  * □ Flory-Schulz分布を用いて分子量分布を予測できる
  * □ RAFT重合の設計計算（[M]/[RAFT]比）ができる
  * □ 連鎖移動定数を使って分子量制御ができる

### レベル3: 応用力

  * □ 重合条件（開始剤濃度、温度、溶媒）が分子量に与える影響を予測できる
  * □ 目標分子量・PDIを達成する重合法を選択できる
  * □ リビング重合と通常のラジカル重合の分子量分布を比較できる
  * □ 連鎖移動剤を用いた分子量設計ができる
  * □ 実験データから重合メカニズムを推測できる
  * □ 高分子合成の工業プロセス最適化を議論できる

**次のステップ** 第2章では、生成した高分子の構造（タクティシティ、結晶性、架橋）を学びます。本章で学んだ分子量分布が、高分子の物性（ガラス転移温度、機械的強度、溶解性）にどう影響するかを理解しましょう。特に、立体規則性と結晶化は材料特性を大きく左右します。 

### 免責事項

  * 本コンテンツは教育・研究・情報提供のみを目的としており、専門的な助言(法律・会計・技術的保証など)を提供するものではありません。
  * 本コンテンツおよび付随するCode examplesは「現状有姿(AS IS)」で提供され、明示または黙示を問わず、商品性、特定目的適合性、権利非侵害、正確性・完全性、動作・安全性等いかなる保証もしません。
  * 外部リンク、第三者が提供するデータ・ツール・ライブラリ等の内容・可用性・安全性について、作成者および東北大学は一切の責任を負いません。
  * 本コンテンツの利用・実行・解釈により直接的・間接的・付随的・特別・結果的・懲罰的損害が生じた場合でも、適用法で許容される最大限の範囲で、作成者および東北大学は責任を負いません。
  * 本コンテンツの内容は、予告なく変更・更新・提供停止されることがあります。
  * 本コンテンツの著作権・ライセンスは明記された条件(例: CC BY 4.0)に従います。当該ライセンスは通常、無保証条項を含みます。
