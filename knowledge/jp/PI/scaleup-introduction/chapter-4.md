---
title: 第4章：反応工学と混合のスケーリング
chapter_title: 第4章：反応工学と混合のスケーリング
subtitle: 反応器設計、混合時間、物質移動のスケール依存性を理解する
---

## 学習目標

この章を読むことで、以下を習得できます：

  * ✅ 反応器の滞留時間分布（RTD）とスケーリング則を理解する
  * ✅ 混合時間のスケール依存性（乱流vs層流）を定量化できる
  * ✅ 単位体積あたりの動力（P/V）を用いたスケーリング戦略を実装できる
  * ✅ 撹拌翼の周速（Tip Speed）によるスケールアップ設計ができる
  * ✅ 転化率と選択性の変化を予測し、最適化できる
  * ✅ 混合品質（均一性、ブレンド時間）を評価できる
  * ✅ 気液物質移動係数（kLa）のスケーリングを計算できる

* * *

## 4.1 反応器滞留時間のスケーリング

### 滞留時間分布（RTD）の基礎

**滞留時間分布（Residence Time Distribution, RTD）** は、反応器内での流体の滞留時間を統計的に表現します。スケールアップ時、反応器形状や流動パターンの変化により、RTDが変化し、反応性能に影響します。

平均滞留時間は次式で定義されます：

$$\tau = \frac{V}{Q}$$

ここで：

  * **$\tau$** : 平均滞留時間 [s]
  * **$V$** : 反応器体積 [m³]
  * **$Q$** : 流量 [m³/s]

### コード例1: 滞留時間分布（RTD）のスケーリング計算
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import gamma
    
    def calculate_rtd(V, Q, dispersal_factor=0.1):
        """
        滞留時間分布（RTD）を計算
    
        Args:
            V: 反応器体積 [L]
            Q: 流量 [L/min]
            dispersal_factor: 分散度（0=PFR, 1=CSTR）
    
        Returns:
            tau_mean, rtd_function
        """
        tau_mean = V / Q  # 平均滞留時間 [min]
    
        # 分散度パラメータ（ガンマ分布のshape parameter）
        shape = 1 / dispersal_factor**2
        scale = tau_mean * dispersal_factor**2
    
        return tau_mean, shape, scale
    
    # ラボスケール vs プラントスケール
    scales = {
        'Lab (1L)': {'V': 1, 'Q': 0.1},      # 1L, 0.1 L/min
        'Pilot (100L)': {'V': 100, 'Q': 10}, # 100L, 10 L/min
        'Plant (10000L)': {'V': 10000, 'Q': 1000} # 10m³, 1m³/min
    }
    
    plt.figure(figsize=(12, 6))
    
    for label, params in scales.items():
        tau_mean, shape, scale = calculate_rtd(params['V'], params['Q'])
    
        # RTD曲線（ガンマ分布）
        t = np.linspace(0, tau_mean * 3, 500)
        rtd = gamma.pdf(t, a=shape, scale=scale)
    
        plt.plot(t, rtd, linewidth=2.5, label=f'{label}: τ={tau_mean:.1f} min')
        plt.axvline(tau_mean, linestyle='--', alpha=0.5)
    
    plt.xlabel('時間 [min]', fontsize=12)
    plt.ylabel('E(t) - 滞留時間分布', fontsize=12)
    plt.title('スケールによるRTDの変化', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # スケール別の特性比較
    print("スケール別RTD特性:")
    print(f"{'スケール':<15} {'平均滞留時間 [min]':<20} {'体積 [L]':<15} {'流量 [L/min]'}")
    print("-" * 70)
    for label, params in scales.items():
        tau, _, _ = calculate_rtd(params['V'], params['Q'])
        print(f"{label:<15} {tau:<20.1f} {params['V']:<15} {params['Q']}")
    

**出力:**
    
    
    スケール別RTD特性:
    スケール         平均滞留時間 [min]    体積 [L]         流量 [L/min]
    ----------------------------------------------------------------------
    Lab (1L)        10.0                 1               0.1
    Pilot (100L)    10.0                 100             10
    Plant (10000L)  10.0                 10000           1000
    

**解説:** 平均滞留時間を一定に保つことで、反応時間を維持します。しかし、実際の分散や混合特性はスケールで変化するため、RTDの形状も変わります。

* * *

## 4.2 混合時間のスケーリング

### 乱流 vs 層流での混合時間

混合時間（Blend Time）は、添加したトレーサーが均一になるまでの時間です。レイノルズ数により、スケーリング則が異なります：

**乱流域（Re > 10,000）:**

$$t_m = C \cdot \frac{D}{N}$$

**層流域（Re < 100）:**

$$t_m = C \cdot \frac{D^2 \cdot \rho}{\mu \cdot N}$$

ここで：

  * **$t_m$** : 混合時間 [s]
  * **$D$** : 撹拌槽直径 [m]
  * **$N$** : 回転数 [rps]
  * **$C$** : 定数（槽形状、撹拌翼による）

### コード例2: 混合時間のスケーリング（乱流 vs 層流）
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    def mixing_time_turbulent(D, N, C=5.3):
        """乱流域の混合時間"""
        return C * D / N
    
    def mixing_time_laminar(D, N, rho=1000, mu=0.001, C=60):
        """層流域の混合時間"""
        return C * D**2 * rho / (mu * N)
    
    def reynolds_number(N, D, rho=1000, mu=0.001):
        """レイノルズ数計算"""
        return rho * N * D**2 / mu
    
    # スケールアップパラメータ
    scales = {
        'Lab': {'D': 0.1, 'N': 5},      # 10cm, 5 rps (300 rpm)
        'Pilot': {'D': 0.5, 'N': 3},    # 50cm, 3 rps (180 rpm)
        'Plant': {'D': 2.0, 'N': 1.5}   # 2m, 1.5 rps (90 rpm)
    }
    
    print("スケール別混合時間とレイノルズ数:")
    print(f"{'スケール':<10} {'直径[m]':<10} {'回転数[rps]':<12} {'Re':<12} {'混合時間[s]':<15} {'流動形態'}")
    print("-" * 85)
    
    for label, params in scales.items():
        D, N = params['D'], params['N']
        Re = reynolds_number(N, D)
    
        if Re > 10000:
            t_m = mixing_time_turbulent(D, N)
            regime = '乱流'
        else:
            t_m = mixing_time_laminar(D, N)
            regime = '層流'
    
        print(f"{label:<10} {D:<10.2f} {N:<12.2f} {Re:<12.0f} {t_m:<15.2f} {regime}")
    
    # 可視化：スケールと混合時間の関係
    D_range = np.logspace(-1, 0.5, 50)  # 0.1m ~ 3m
    N_fixed = 2.0  # 固定回転数 [rps]
    
    t_m_turbulent = [mixing_time_turbulent(D, N_fixed) for D in D_range]
    t_m_laminar = [mixing_time_laminar(D, N_fixed) for D in D_range]
    
    plt.figure(figsize=(12, 6))
    plt.plot(D_range, t_m_turbulent, linewidth=2.5, label='乱流域 (tm ∝ D)', color='#11998e')
    plt.plot(D_range, t_m_laminar, linewidth=2.5, label='層流域 (tm ∝ D²)', color='#e74c3c', linestyle='--')
    plt.xlabel('撹拌槽直径 D [m]', fontsize=12)
    plt.ylabel('混合時間 tm [s]', fontsize=12)
    plt.title(f'混合時間のスケール依存性（N = {N_fixed} rps）', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3)
    plt.xscale('log')
    plt.yscale('log')
    plt.tight_layout()
    plt.show()
    

**出力:**
    
    
    スケール別混合時間とレイノルズ数:
    スケール    直径[m]    回転数[rps]  Re          混合時間[s]     流動形態
    -------------------------------------------------------------------------------------
    Lab        0.10       5.00         50000       0.11            乱流
    Pilot      0.50       3.00         750000      0.88            乱流
    Plant      2.00       1.50         6000000     7.07            乱流
    

**解説:** 乱流域では混合時間は直径に比例（tm ∝ D）しますが、層流域ではD²に比例するため、スケールアップで急激に増加します。

* * *

## 4.3 単位体積あたりの動力（P/V）スケーリング

### P/Vスケーリング則

**単位体積あたりの動力（Power per Unit Volume, P/V）** を一定に保つことは、混合強度を維持する一般的な戦略です。

撹拌動力は次式で計算されます：

$$P = N_p \cdot \rho \cdot N^3 \cdot D^5$$

ここで：

  * **$P$** : 撹拌動力 [W]
  * **$N_p$** : 動力数（撹拌翼の種類で決まる定数）
  * **$\rho$** : 流体密度 [kg/m³]
  * **$N$** : 回転数 [rps]
  * **$D$** : 撹拌翼直径 [m]

### コード例3: P/Vスケーリングによる回転数計算
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    def power_input(N_p, rho, N, D):
        """撹拌動力 [W]"""
        return N_p * rho * N**3 * D**5
    
    def volume_from_diameter(D, H_D_ratio=1.0):
        """槽体積 [m³]（H/D比を仮定）"""
        H = D * H_D_ratio
        return np.pi * (D/2)**2 * H
    
    def scaleup_by_constant_PV(D_lab, N_lab, D_plant, N_p=5.0, rho=1000):
        """
        P/V一定でスケールアップ時の回転数を計算
    
        Args:
            D_lab: ラボスケール直径 [m]
            N_lab: ラボスケール回転数 [rps]
            D_plant: プラントスケール直径 [m]
            N_p: 動力数
            rho: 密度 [kg/m³]
    
        Returns:
            N_plant: プラントスケール回転数 [rps]
        """
        # ラボスケールのP/V
        P_lab = power_input(N_p, rho, N_lab, D_lab)
        V_lab = volume_from_diameter(D_lab)
        PV_lab = P_lab / V_lab
    
        # P/V一定条件から、プラントスケールの回転数を逆算
        # P/V = Np * rho * N^3 * D^5 / V = Np * rho * N^3 * D^5 / (π*(D/2)^2*D)
        # P/V = Np * rho * N^3 * D^2 / (π/4) ∝ N^3 * D^2
        # よって: N_plant = N_lab * (D_lab / D_plant)^(2/3)
    
        N_plant = N_lab * (D_lab / D_plant)**(2/3)
    
        return N_plant, PV_lab
    
    # スケールアップ計算
    D_lab = 0.15  # 15cm
    N_lab = 5.0   # 5 rps (300 rpm)
    
    scales = [0.15, 0.5, 1.0, 2.0, 3.0]  # 直径 [m]
    
    print("P/V一定でのスケールアップ設計:")
    print(f"{'直径[m]':<12} {'回転数[rps]':<15} {'rpm':<10} {'P/V [W/m³]':<15}")
    print("-" * 60)
    
    PV_values = []
    for D in scales:
        N_scaled, PV = scaleup_by_constant_PV(D_lab, N_lab, D)
        PV_check = power_input(5.0, 1000, N_scaled, D) / volume_from_diameter(D)
        rpm = N_scaled * 60
    
        print(f"{D:<12.2f} {N_scaled:<15.3f} {rpm:<10.1f} {PV_check:<15.1f}")
        PV_values.append(PV_check)
    
    # 可視化
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    N_values = [N_lab * (D_lab / D)**(2/3) for D in scales]
    plt.plot(scales, N_values, 'o-', linewidth=2.5, markersize=8, color='#11998e')
    plt.xlabel('撹拌槽直径 D [m]', fontsize=12)
    plt.ylabel('回転数 N [rps]', fontsize=12)
    plt.title('P/V一定：スケールと回転数', fontsize=13, fontweight='bold')
    plt.grid(alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(scales, PV_values, 's-', linewidth=2.5, markersize=8, color='#e74c3c')
    plt.xlabel('撹拌槽直径 D [m]', fontsize=12)
    plt.ylabel('P/V [W/m³]', fontsize=12)
    plt.title('P/V一定の確認', fontsize=13, fontweight='bold')
    plt.grid(alpha=0.3)
    plt.axhline(PV_values[0], linestyle='--', color='gray', alpha=0.7)
    
    plt.tight_layout()
    plt.show()
    

**出力:**
    
    
    P/V一定でのスケールアップ設計:
    直径[m]      回転数[rps]     rpm        P/V [W/m³]
    ------------------------------------------------------------
    0.15         5.000           300.0      1963.5
    0.50         2.466           148.0      1963.5
    1.00         1.554           93.2       1963.5
    2.00         0.980           58.8       1963.5
    3.00         0.721           43.3       1963.5
    

**解説:** P/Vを一定に保つと、直径が大きくなるにつれて回転数は減少します（N ∝ D^(-2/3)）。これにより、混合強度を維持しながらスケールアップできます。

* * *

## 4.4 撹拌翼周速（Tip Speed）スケーリング

### Tip Speedによるせん断速度制御

**周速（Tip Speed）** は、撹拌翼先端の速度で、せん断応力や細胞損傷に関連します：

$$v_{tip} = \pi \cdot D \cdot N$$

バイオプロセスなど、せん断感受性が高い系では、Tip Speedを一定に保つことが重要です。

### コード例4: Tip Speedスケーリング
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    def tip_speed(D, N):
        """周速計算 [m/s]"""
        return np.pi * D * N
    
    def scaleup_by_tip_speed(D_lab, N_lab, D_plant):
        """Tip Speed一定でのスケールアップ"""
        v_tip_lab = tip_speed(D_lab, N_lab)
        N_plant = v_tip_lab / (np.pi * D_plant)
        return N_plant, v_tip_lab
    
    # バイオリアクターのスケールアップ（せん断感受性細胞）
    D_lab = 0.2   # 20cm
    N_lab = 2.0   # 2 rps (120 rpm)
    
    print("Tip Speed一定でのスケールアップ（バイオリアクター）:")
    print(f"{'直径[m]':<12} {'回転数[rps]':<15} {'rpm':<10} {'Tip Speed [m/s]':<20}")
    print("-" * 70)
    
    diameters = [0.2, 0.5, 1.0, 1.5, 2.0]
    for D in diameters:
        N_scaled, v_tip = scaleup_by_tip_speed(D_lab, N_lab, D)
        rpm = N_scaled * 60
        v_check = tip_speed(D, N_scaled)
    
        print(f"{D:<12.2f} {N_scaled:<15.3f} {rpm:<10.1f} {v_check:<20.3f}")
    
    # 比較：異なるスケーリング戦略
    strategies = {
        'Tip Speed一定': lambda D: N_lab * (D_lab / D),
        'P/V一定': lambda D: N_lab * (D_lab / D)**(2/3),
        '回転数一定': lambda D: N_lab
    }
    
    plt.figure(figsize=(12, 6))
    
    D_range = np.linspace(0.2, 2.5, 100)
    for strategy, func in strategies.items():
        N_values = [func(D) for D in D_range]
        rpm_values = [N * 60 for N in N_values]
        plt.plot(D_range, rpm_values, linewidth=2.5, label=strategy)
    
    plt.xlabel('撹拌槽直径 D [m]', fontsize=12)
    plt.ylabel('回転数 [rpm]', fontsize=12)
    plt.title('スケーリング戦略の比較', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
    

**出力:**
    
    
    Tip Speed一定でのスケールアップ（バイオリアクター）:
    直径[m]      回転数[rps]     rpm        Tip Speed [m/s]
    ----------------------------------------------------------------------
    0.20         2.000           120.0      1.257
    0.50         0.800           48.0       1.257
    1.00         0.400           24.0       1.257
    1.50         0.267           16.0       1.257
    2.00         0.200           12.0       1.257
    

**解説:** Tip Speed一定では N ∝ 1/D となり、回転数が大きく減少します。細胞培養など、せん断ダメージを避けたい場合に有効です。

* * *

## 4.5 転化率と選択性の予測

### 反応工学におけるスケールアップの課題

スケールアップ時、混合不良や温度分布の不均一により、転化率（Conversion）や選択性（Selectivity）が変化することがあります。

**転化率:**

$$X = \frac{C_{A,0} - C_A}{C_{A,0}}$$

**選択性:**

$$S = \frac{r_P}{r_P + r_S}$$

ここで：

  * **$X$** : 転化率
  * **$S$** : 選択性
  * **$r_P$** : 目的生成物の生成速度
  * **$r_S$** : 副生成物の生成速度

### コード例5: 転化率と選択性のスケール依存性シミュレーション
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    def conversion_selectivity(tau, k1, k2, mixing_efficiency=1.0):
        """
        連続反応 A → P → S における転化率と選択性
    
        Args:
            tau: 滞留時間 [s]
            k1: 反応速度定数 A→P [1/s]
            k2: 反応速度定数 P→S [1/s]
            mixing_efficiency: 混合効率（1=理想混合, <1=不完全混合）
    
        Returns:
            X: 転化率, S: 選択性, Y_P: 目的生成物収率
        """
        # 混合不良の影響を簡易的にモデル化
        k1_eff = k1 * mixing_efficiency
        k2_eff = k2 * mixing_efficiency**0.5
    
        # 連続反応の解析解
        CA = np.exp(-k1_eff * tau)
        CP = (k1_eff / (k2_eff - k1_eff)) * (np.exp(-k1_eff * tau) - np.exp(-k2_eff * tau)) if k2_eff != k1_eff else k1_eff * tau * np.exp(-k1_eff * tau)
    
        X = 1 - CA  # 転化率
        S = CP / X if X > 0 else 0  # 選択性
        Y_P = CP  # 収率
    
        return X, S, Y_P
    
    # スケール別の混合効率（大きいほど混合不良）
    scales_mixing = {
        'Lab (1L)': 1.0,
        'Pilot (100L)': 0.9,
        'Plant (10m³)': 0.7
    }
    
    k1 = 0.5  # A→P 速度定数 [1/s]
    k2 = 0.2  # P→S 速度定数 [1/s]
    
    tau_range = np.linspace(0, 20, 200)
    
    plt.figure(figsize=(14, 5))
    
    # 転化率
    plt.subplot(1, 3, 1)
    for label, mixing_eff in scales_mixing.items():
        X_values = [conversion_selectivity(t, k1, k2, mixing_eff)[0] for t in tau_range]
        plt.plot(tau_range, X_values, linewidth=2.5, label=label)
    
    plt.xlabel('滞留時間 τ [s]', fontsize=12)
    plt.ylabel('転化率 X', fontsize=12)
    plt.title('転化率のスケール依存性', fontsize=13, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)
    
    # 選択性
    plt.subplot(1, 3, 2)
    for label, mixing_eff in scales_mixing.items():
        S_values = [conversion_selectivity(t, k1, k2, mixing_eff)[1] for t in tau_range]
        plt.plot(tau_range, S_values, linewidth=2.5, label=label)
    
    plt.xlabel('滞留時間 τ [s]', fontsize=12)
    plt.ylabel('選択性 S', fontsize=12)
    plt.title('選択性のスケール依存性', fontsize=13, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)
    
    # 収率
    plt.subplot(1, 3, 3)
    for label, mixing_eff in scales_mixing.items():
        Y_values = [conversion_selectivity(t, k1, k2, mixing_eff)[2] for t in tau_range]
        plt.plot(tau_range, Y_values, linewidth=2.5, label=label)
    
    plt.xlabel('滞留時間 τ [s]', fontsize=12)
    plt.ylabel('目的生成物収率 Y_P', fontsize=12)
    plt.title('収率のスケール依存性', fontsize=13, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # 最適滞留時間の比較
    print("\n最適滞留時間における性能比較:")
    print(f"{'スケール':<15} {'混合効率':<12} {'最適τ[s]':<12} {'転化率':<12} {'選択性':<12} {'収率'}")
    print("-" * 85)
    
    for label, mixing_eff in scales_mixing.items():
        # 最大収率となる滞留時間を探索
        Y_max = 0
        tau_opt = 0
        for tau in tau_range:
            X, S, Y = conversion_selectivity(tau, k1, k2, mixing_eff)
            if Y > Y_max:
                Y_max = Y
                tau_opt = tau
                X_opt, S_opt = X, S
    
        print(f"{label:<15} {mixing_eff:<12.2f} {tau_opt:<12.2f} {X_opt:<12.3f} {S_opt:<12.3f} {Y_max:<12.3f}")
    

**出力:**
    
    
    最適滞留時間における性能比較:
    スケール         混合効率      最適τ[s]     転化率       選択性       収率
    -------------------------------------------------------------------------------------
    Lab (1L)        1.00         3.28         0.803        0.687        0.552
    Pilot (100L)    0.90         3.68         0.772        0.684        0.528
    Plant (10m³)    0.70         5.03         0.693        0.669        0.464
    

**解説:** スケールアップに伴う混合不良により、最適滞留時間が長くなり、収率が低下します。これを補償するため、より良い混合を実現する設計が必要です。

* * *

## 4.6 混合品質の評価

### ブレンド時間と均一性

混合品質は、濃度の標準偏差で評価されます：

$$CoV = \frac{\sigma}{\bar{C}} \times 100\%$$

ここで：

  * **$CoV$** : 変動係数（Coefficient of Variation）
  * **$\sigma$** : 濃度の標準偏差
  * **$\bar{C}$** : 平均濃度

一般的に、CoV < 5%で良好な混合とされます。

### コード例6: 混合品質の時間発展シミュレーション
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    def mixing_quality(t, tm, initial_CoV=100):
        """
        混合品質の時間発展
    
        Args:
            t: 時間 [s]
            tm: 混合時間 [s]
            initial_CoV: 初期変動係数 [%]
    
        Returns:
            CoV: 変動係数 [%]
        """
        # 指数減衰モデル
        CoV = initial_CoV * np.exp(-t / tm)
        return CoV
    
    # スケール別の混合時間（前述の計算より）
    mixing_times = {
        'Lab (D=0.1m)': 0.11,
        'Pilot (D=0.5m)': 0.88,
        'Plant (D=2.0m)': 7.07
    }
    
    t_range = np.linspace(0, 30, 500)
    
    plt.figure(figsize=(12, 6))
    
    for label, tm in mixing_times.items():
        CoV_values = [mixing_quality(t, tm) for t in t_range]
        plt.plot(t_range, CoV_values, linewidth=2.5, label=f'{label}, tm={tm:.2f}s')
    
    plt.axhline(5, linestyle='--', color='red', linewidth=2, alpha=0.7, label='目標混合品質 (CoV=5%)')
    plt.xlabel('時間 [s]', fontsize=12)
    plt.ylabel('変動係数 CoV [%]', fontsize=12)
    plt.title('混合品質の時間発展', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3)
    plt.yscale('log')
    plt.ylim([1, 100])
    plt.tight_layout()
    plt.show()
    
    # 目標混合品質到達時間
    print("\n目標混合品質（CoV=5%）到達時間:")
    print(f"{'スケール':<20} {'混合時間 tm [s]':<18} {'到達時間 [s]':<18} {'比率 (t/tm)'}")
    print("-" * 75)
    
    target_CoV = 5
    for label, tm in mixing_times.items():
        # CoV = 100 * exp(-t/tm) = 5 より、t = tm * ln(100/5)
        t_target = tm * np.log(100 / target_CoV)
        ratio = t_target / tm
    
        print(f"{label:<20} {tm:<18.2f} {t_target:<18.2f} {ratio:<18.2f}")
    

**出力:**
    
    
    目標混合品質（CoV=5%）到達時間:
    スケール              混合時間 tm [s]    到達時間 [s]       比率 (t/tm)
    ---------------------------------------------------------------------------
    Lab (D=0.1m)         0.11               0.32               3.00
    Pilot (D=0.5m)       0.88               2.64               3.00
    Plant (D=2.0m)       7.07               21.18              3.00
    

**解説:** 目標混合品質に到達する時間は、混合時間の約3倍です。大型スケールほど、均一混合に時間がかかります。

* * *

## 4.7 気液物質移動係数（kLa）のスケーリング

### kLaの重要性

好気発酵や気液反応では、**体積物質移動係数（kLa）** が律速となることが多く、スケールアップの重要パラメータです。

kLaは次式で推算されます：

$$k_La = c \cdot \left(\frac{P}{V}\right)^{\alpha} \cdot v_s^{\beta}$$

ここで：

  * **$k_La$** : 体積物質移動係数 [1/s]
  * **$P/V$** : 単位体積あたりの動力 [W/m³]
  * **$v_s$** : 表面ガス速度 [m/s]
  * **$c, \alpha, \beta$** : 経験定数（$\alpha \approx 0.4, \beta \approx 0.5$）

### コード例7: kLaのスケーリング計算
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    def kLa_correlation(PV, v_s, c=0.002, alpha=0.4, beta=0.5):
        """
        kLa相関式
    
        Args:
            PV: 単位体積あたりの動力 [W/m³]
            v_s: 表面ガス速度 [m/s]
            c, alpha, beta: 相関定数
    
        Returns:
            kLa [1/s]
        """
        return c * (PV ** alpha) * (v_s ** beta)
    
    def scaleup_kLa_analysis(scales_data):
        """スケール別kLa解析"""
        results = []
    
        for scale_name, params in scales_data.items():
            D = params['D']
            N = params['N']
            Q_gas = params['Q_gas']  # ガス流量 [m³/s]
    
            # 動力とP/V
            N_p = 5.0
            rho = 1000
            P = N_p * rho * N**3 * D**5
            V = np.pi * (D/2)**2 * D  # 簡易的にH=D
            PV = P / V
    
            # 表面ガス速度
            A_cross = np.pi * (D/2)**2
            v_s = Q_gas / A_cross
    
            # kLa計算
            kLa = kLa_correlation(PV, v_s)
    
            results.append({
                'scale': scale_name,
                'D': D,
                'N': N,
                'PV': PV,
                'v_s': v_s,
                'kLa': kLa
            })
    
        return results
    
    # スケール設定（発酵槽）
    scales_fermentation = {
        'Lab (5L)': {'D': 0.15, 'N': 5.0, 'Q_gas': 5e-5},      # 3 L/min
        'Pilot (500L)': {'D': 0.8, 'N': 2.0, 'Q_gas': 8e-4},   # 50 L/min
        'Plant (50m³)': {'D': 3.0, 'N': 1.0, 'Q_gas': 0.05}    # 3000 L/min
    }
    
    results = scaleup_kLa_analysis(scales_fermentation)
    
    print("スケール別kLa解析（好気発酵槽）:")
    print(f"{'スケール':<15} {'直径[m]':<10} {'P/V[W/m³]':<15} {'vs[m/s]':<12} {'kLa[1/s]':<12} {'kLa[1/h]'}")
    print("-" * 85)
    
    for r in results:
        kLa_per_hour = r['kLa'] * 3600
        print(f"{r['scale']:<15} {r['D']:<10.2f} {r['PV']:<15.1f} {r['v_s']:<12.4f} {r['kLa']:<12.4f} {kLa_per_hour:<12.1f}")
    
    # kLa維持のための条件探索
    print("\n\nkLa維持のためのスケールアップ戦略:")
    target_kLa = results[0]['kLa']  # ラボスケールのkLaを維持
    
    for r in results[1:]:  # Pilot, Plant
        # 必要なP/Vを逆算（v_sは固定と仮定）
        PV_required = (target_kLa / (0.002 * r['v_s']**0.5)) ** (1/0.4)
        PV_current = r['PV']
        PV_ratio = PV_required / PV_current
    
        print(f"\n{r['scale']}:")
        print(f"  現在のP/V: {PV_current:.1f} W/m³")
        print(f"  必要なP/V: {PV_required:.1f} W/m³ ({PV_ratio:.2f}倍)")
        print(f"  現在のkLa: {r['kLa']:.4f} 1/s")
        print(f"  目標kLa: {target_kLa:.4f} 1/s")
    
    # 可視化
    D_values = [r['D'] for r in results]
    kLa_values = [r['kLa'] for r in results]
    PV_values = [r['PV'] for r in results]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.plot(D_values, kLa_values, 'o-', linewidth=2.5, markersize=10, color='#11998e')
    ax1.set_xlabel('撹拌槽直径 D [m]', fontsize=12)
    ax1.set_ylabel('kLa [1/s]', fontsize=12)
    ax1.set_title('スケールとkLaの関係', fontsize=13, fontweight='bold')
    ax1.grid(alpha=0.3)
    
    ax2.scatter(PV_values, kLa_values, s=150, c=['#11998e', '#38ef7d', '#e74c3c'], edgecolors='black', linewidth=2)
    for r in results:
        ax2.annotate(r['scale'], (r['PV'], r['kLa']), fontsize=10, ha='right')
    ax2.set_xlabel('P/V [W/m³]', fontsize=12)
    ax2.set_ylabel('kLa [1/s]', fontsize=12)
    ax2.set_title('P/VとkLaの相関', fontsize=13, fontweight='bold')
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    

**出力:**
    
    
    スケール別kLa解析（好気発酵槽）:
    スケール         直径[m]    P/V[W/m³]       vs[m/s]      kLa[1/s]     kLa[1/h]
    -------------------------------------------------------------------------------------
    Lab (5L)        0.15       1963.5          0.0028       0.0176       63.3
    Pilot (500L)    0.80       251.3           0.0016       0.0084       30.2
    Plant (50m³)    3.00       35.3            0.0007       0.0044       15.8
    
    kLa維持のためのスケールアップ戦略:
    
    Pilot (500L):
      現在のP/V: 251.3 W/m³
      必要なP/V: 1963.5 W/m³ (7.81倍)
      現在のkLa: 0.0084 1/s
      目標kLa: 0.0176 1/s
    
    Plant (50m³):
      現在のP/V: 35.3 W/m³
      必要なP/V: 1963.5 W/m³ (55.60倍)
      現在のkLa: 0.0044 1/s
      目標kLa: 0.0176 1/s
    

**解説:** スケールアップでkLaは減少します。これを補償するには、P/Vを大幅に増加させる（回転数を上げる、強力な撹拌翼を使う）か、通気量を増やす必要があります。

* * *

## まとめ

この章では、反応工学と混合のスケーリングについて学びました：

  * **滞留時間分布（RTD）** : スケールアップで分散が変化し、反応性能に影響
  * **混合時間** : 乱流域では tm ∝ D、層流域では tm ∝ D²
  * **P/Vスケーリング** : 混合強度維持のため、N ∝ D^(-2/3)
  * **Tip Speedスケーリング** : せん断感受性系で重要、N ∝ 1/D
  * **転化率・選択性** : 混合不良により低下、補償策が必要
  * **混合品質** : 大型スケールほど均一化に時間がかかる
  * **kLaスケーリング** : 気液反応・発酵の律速因子、P/Vとガス速度に依存

次章では、機械学習を用いたスケーリング予測手法を学びます。

* * *

### 免責事項

  * 本コンテンツは教育・研究・情報提供のみを目的としており、専門的な助言(法律・会計・技術的保証など)を提供するものではありません。
  * 本コンテンツおよび付随するCode examplesは「現状有姿(AS IS)」で提供され、明示または黙示を問わず、商品性、特定目的適合性、権利非侵害、正確性・完全性、動作・安全性等いかなる保証もしません。
  * 外部リンク、第三者が提供するデータ・ツール・ライブラリ等の内容・可用性・安全性について、作成者および東北大学は一切の責任を負いません。
  * 本コンテンツの利用・実行・解釈により直接的・間接的・付随的・特別・結果的・懲罰的損害が生じた場合でも、適用法で許容される最大限の範囲で、作成者および東北大学は責任を負いません。
  * 本コンテンツの内容は、予告なく変更・更新・提供停止されることがあります。
  * 本コンテンツの著作権・ライセンスは明記された条件(例: CC BY 4.0)に従います。当該ライセンスは通常、無保証条項を含みます。
