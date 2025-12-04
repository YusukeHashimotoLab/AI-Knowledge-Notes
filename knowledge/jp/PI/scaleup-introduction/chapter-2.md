---
title: 第2章：相似則と無次元数
chapter_title: 第2章：相似則と無次元数
subtitle: スケーリングの基礎となる物理的相似性の理解
---

## 学習目標

この章を読むことで、以下を習得できます：

  * ✅ 相似則（幾何学的、運動学的、動力学的相似）の原理を理解する
  * ✅ 主要な無次元数（Re, Fr, We, Po等）の物理的意味を説明できる
  * ✅ Pythonで無次元数を計算し、流動状態を判定できる
  * ✅ 複数の無次元数を用いた多基準相似解析を実行できる
  * ✅ スケールアップ時の相似則適用の限界を理解する

* * *

## 2.1 相似則の基礎

### 3つの相似性

スケールアップ・スケールダウンを成功させるためには、小型装置と大型装置の間で**相似性（Similarity）** を保つ必要があります。相似性には3つの階層があります：

相似性の種類 | 定義 | スケーリングへの影響  
---|---|---  
**幾何学的相似** | 形状・寸法比が一定 | 全ての長さが同じ比率でスケール（L₂/L₁ = S）  
**運動学的相似** | 速度場の形状が相似 | 対応する点で速度の方向と比率が同じ  
**動力学的相似** | 力の比率が相似 | 慣性力、粘性力、重力等の比率が一定  
  
**重要:** 動力学的相似を達成するには、関連する無次元数を一致させる必要があります。
    
    
    ```mermaid
    graph TD
        A[幾何学的相似形状を同じに] --> B[運動学的相似流れのパターンを同じに]
        B --> C[動力学的相似力のバランスを同じに]
        C --> D[プロセス性能の相似反応・分離性能の再現]
    
        style A fill:#e3f2fd
        style B fill:#fff3e0
        style C fill:#e8f5e9
        style D fill:#f3e5f5
    ```

### 無次元数とは

**無次元数（Dimensionless Number）** は、異なる物理的力や現象の比を表す無次元のパラメータです。無次元数が同じであれば、スケールが異なっても同じ物理現象が生じます。

一般形：

$$ \text{無次元数} = \frac{\text{ある種の力}}{\text{別の種類の力}} = \frac{\text{特性時間}_1}{\text{特性時間}_2} $$

* * *

## 2.2 流体力学の無次元数

### Reynolds数（レイノルズ数）

Reynolds数（Re）は、**慣性力** と**粘性力** の比を表します：

$$ \text{Re} = \frac{\rho u L}{\mu} = \frac{uL}{\nu} $$

ここで：

  * $\rho$: 密度 [kg/m³]
  * $u$: 代表速度 [m/s]
  * $L$: 代表長さ [m]（管径、攪拌翼径等）
  * $\mu$: 粘度 [Pa·s]
  * $\nu = \mu/\rho$: 動粘度 [m²/s]

**物理的意味:**

  * Re << 1: 粘性力支配（層流、クリープ流）
  * Re ≈ 2,300（管流）または 10,000（攪拌槽）: 遷移領域
  * Re >> 1: 慣性力支配（乱流）

### コード例1: Reynolds数の計算と流動状態の判定
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    def reynolds_number(rho, u, L, mu):
        """
        Reynolds数を計算
    
        Parameters:
        -----------
        rho : float
            流体密度 [kg/m³]
        u : float
            代表速度 [m/s]
        L : float
            代表長さ [m]
        mu : float
            粘度 [Pa·s]
    
        Returns:
        --------
        Re : float
            Reynolds数 [-]
        """
        return rho * u * L / mu
    
    def flow_regime(Re, system_type='pipe'):
        """
        Reynolds数から流動状態を判定
    
        Parameters:
        -----------
        Re : float
            Reynolds数
        system_type : str
            'pipe'（管流）または 'stirred'（攪拌槽）
    
        Returns:
        --------
        regime : str
            流動状態（'Laminar', 'Transition', 'Turbulent'）
        """
        if system_type == 'pipe':
            Re_crit = 2300
        elif system_type == 'stirred':
            Re_crit = 10000
        else:
            Re_crit = 2300  # デフォルト
    
        if Re < Re_crit:
            return 'Laminar (層流)'
        elif Re < Re_crit * 2:
            return 'Transition (遷移)'
        else:
            return 'Turbulent (乱流)'
    
    # 実例：水（20°C）の管内流
    # 物性値
    rho_water = 998.2  # kg/m³
    mu_water = 1.002e-3  # Pa·s
    
    # パイプライン条件
    pipe_diameter = np.array([0.025, 0.05, 0.1, 0.2])  # m (25mm, 50mm, 100mm, 200mm)
    flow_velocity = 1.5  # m/s
    
    print("=" * 70)
    print("Reynolds数計算と流動状態判定（水の管内流）")
    print("=" * 70)
    print(f"流体: 水（20°C）、密度 = {rho_water} kg/m³、粘度 = {mu_water*1000:.3f} mPa·s")
    print(f"流速: {flow_velocity} m/s")
    print("-" * 70)
    
    for D in pipe_diameter:
        Re = reynolds_number(rho_water, flow_velocity, D, mu_water)
        regime = flow_regime(Re, 'pipe')
        print(f"管径 {D*1000:6.1f} mm → Re = {Re:10,.0f} → {regime}")
    
    # 可視化：Reynolds数 vs. 管径
    Re_values = reynolds_number(rho_water, flow_velocity, pipe_diameter, mu_water)
    
    plt.figure(figsize=(10, 6))
    plt.plot(pipe_diameter * 1000, Re_values, 'o-', linewidth=2.5,
             markersize=10, color='#11998e', label='Reynolds数')
    plt.axhline(y=2300, color='orange', linestyle='--', linewidth=2,
                label='層流/乱流境界 (Re = 2,300)')
    plt.axhline(y=4000, color='red', linestyle=':', linewidth=2,
                label='完全乱流領域 (Re ≈ 4,000)')
    plt.xlabel('管径 [mm]', fontsize=12, fontweight='bold')
    plt.ylabel('Reynolds数 [-]', fontsize=12, fontweight='bold')
    plt.title('管径とReynolds数の関係（水、流速 1.5 m/s）',
              fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3)
    plt.yscale('log')
    plt.tight_layout()
    plt.show()
    

**出力:**
    
    
    ======================================================================
    Reynolds数計算と流動状態判定（水の管内流）
    ======================================================================
    流体: 水（20°C）、密度 = 998.2 kg/m³、粘度 = 1.002 mPa·s
    流速: 1.5 m/s
    ----------------------------------------------------------------------
    管径   25.0 mm → Re =     37,365 → Turbulent (乱流)
    管径   50.0 mm → Re =     74,730 → Turbulent (乱流)
    管径  100.0 mm → Re =    149,460 → Turbulent (乱流)
    管径  200.0 mm → Re =    298,920 → Turbulent (乱流)
    

**解説:** 水の場合、通常の流速（1-3 m/s）では、比較的小さな管径でも乱流となります。これはスケールアップ時に乱流状態を維持しやすいことを意味します。

* * *

### Froude数（フルード数）

Froude数（Fr）は、**慣性力** と**重力** の比を表します：

$$ \text{Fr} = \frac{u}{\sqrt{gL}} $$

ここで：

  * $g$: 重力加速度 [m/s²]（≈ 9.81 m/s²）

**重要な系:**

  * 自由表面流（開水路、タンク内の液面変動）
  * 攪拌槽（液面の渦形成）
  * 気液二相流（気泡上昇、スラグ流）

### コード例2: Froude数の計算と自由表面流の評価
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    def froude_number(u, L, g=9.81):
        """
        Froude数を計算
    
        Parameters:
        -----------
        u : float or array
            代表速度 [m/s]
        L : float
            代表長さ [m]（水深、攪拌槽直径等）
        g : float
            重力加速度 [m/s²]（デフォルト: 9.81）
    
        Returns:
        --------
        Fr : float or array
            Froude数 [-]
        """
        return u / np.sqrt(g * L)
    
    def flow_type_froude(Fr):
        """
        Froude数から流れの型を判定
    
        Parameters:
        -----------
        Fr : float
            Froude数
    
        Returns:
        --------
        flow_type : str
            流れの型
        """
        if Fr < 1:
            return 'Subcritical (常流, 重力支配)'
        elif Fr == 1:
            return 'Critical (限界流)'
        else:
            return 'Supercritical (射流, 慣性支配)'
    
    # 実例：攪拌槽のスケールアップ（Froude数一定）
    # ラボスケール
    D_lab = 0.1  # m (10 cm 直径)
    N_lab = 5.0  # rps (回転数)
    u_lab = np.pi * D_lab * N_lab  # 周速度
    
    Fr_lab = froude_number(u_lab, D_lab)
    
    print("=" * 70)
    print("Froude数一定でのスケールアップ（攪拌槽）")
    print("=" * 70)
    print(f"ラボスケール: 直径 = {D_lab*100:.1f} cm, 回転数 = {N_lab:.2f} rps")
    print(f"周速度 = {u_lab:.3f} m/s, Froude数 = {Fr_lab:.3f}")
    print(f"流れの型: {flow_type_froude(Fr_lab)}")
    print("-" * 70)
    
    # パイロットおよび工業スケール
    scale_factors = np.array([1, 5, 10, 20])  # スケール倍率
    D_scale = D_lab * scale_factors
    N_scale = N_lab / np.sqrt(scale_factors)  # Froude数一定の条件
    u_scale = np.pi * D_scale * N_scale
    
    print("\nスケールアップ結果（Froude数一定）:")
    print("-" * 70)
    for i, S in enumerate(scale_factors):
        print(f"スケール倍率 {S:2.0f}x → 直径 {D_scale[i]*100:6.1f} cm, "
              f"回転数 {N_scale[i]:.3f} rps → Fr = {froude_number(u_scale[i], D_scale[i]):.3f}")
    
    # 可視化
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # 左図：回転数の変化
    ax1.plot(scale_factors, N_scale, 'o-', linewidth=2.5, markersize=10,
             color='#11998e', label='Froude数一定')
    ax1.axhline(y=N_lab, color='red', linestyle='--', linewidth=2,
                label=f'ラボスケール (N = {N_lab:.2f} rps)')
    ax1.set_xlabel('スケール倍率 [-]', fontsize=12, fontweight='bold')
    ax1.set_ylabel('回転数 [rps]', fontsize=12, fontweight='bold')
    ax1.set_title('Froude数一定スケーリング: 回転数の変化', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(alpha=0.3)
    
    # 右図：動力数への影響
    Power_relative = scale_factors**2.5  # N³D⁵ ∝ S²·⁵ (Fr一定)
    ax2.plot(scale_factors, Power_relative, 's-', linewidth=2.5, markersize=10,
             color='#e74c3c', label='相対動力（S^2.5則）')
    ax2.set_xlabel('スケール倍率 [-]', fontsize=12, fontweight='bold')
    ax2.set_ylabel('相対動力 [-]', fontsize=12, fontweight='bold')
    ax2.set_title('Froude数一定時の動力増加', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(alpha=0.3)
    ax2.set_yscale('log')
    
    plt.tight_layout()
    plt.show()
    

**出力:**
    
    
    ======================================================================
    Froude数一定でのスケールアップ（攪拌槽）
    ======================================================================
    ラボスケール: 直径 = 10.0 cm, 回転数 = 5.00 rps
    周速度 = 1.571 m/s, Froude数 = 1.597
    流れの型: Supercritical (射流, 慣性支配)
    ----------------------------------------------------------------------
    
    スケールアップ結果（Froude数一定）:
    ----------------------------------------------------------------------
    スケール倍率  1x → 直径   10.0 cm, 回転数 5.000 rps → Fr = 1.597
    スケール倍率  5x → 直径   50.0 cm, 回転数 2.236 rps → Fr = 1.597
    スケール倍率 10x → 直径  100.0 cm, 回転数 1.581 rps → Fr = 1.597
    スケール倍率 20x → 直径  200.0 cm, 回転数 1.118 rps → Fr = 1.597
    

**解説:** Froude数を一定に保つと、回転数はスケール倍率の平方根に反比例します（$N \propto S^{-0.5}$）。これは自由表面の渦形成を相似に保つのに有効ですが、動力は$S^{2.5}$で増加します。

* * *

### Weber数（ウェーバー数）

Weber数（We）は、**慣性力** と**表面張力** の比を表します：

$$ \text{We} = \frac{\rho u^2 L}{\sigma} $$

ここで：

  * $\sigma$: 表面張力 [N/m]

**重要な系:**

  * 液滴・気泡形成
  * 噴霧・アトマイゼーション
  * 気液界面を持つ二相流

### コード例3: Weber数の計算と液滴分裂の評価
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    def weber_number(rho, u, L, sigma):
        """
        Weber数を計算
    
        Parameters:
        -----------
        rho : float
            流体密度 [kg/m³]
        u : float
            相対速度 [m/s]
        L : float
            代表長さ（液滴径等） [m]
        sigma : float
            表面張力 [N/m]
    
        Returns:
        --------
        We : float
            Weber数 [-]
        """
        return rho * u**2 * L / sigma
    
    def droplet_regime(We):
        """
        Weber数から液滴の状態を判定
    
        Parameters:
        -----------
        We : float
            Weber数
    
        Returns:
        --------
        regime : str
            液滴の状態
        """
        if We < 1:
            return '安定（表面張力支配）'
        elif We < 12:
            return '変形開始'
        elif We < 100:
            return 'バッグ分裂'
        else:
            return '微細化（カタストロフィック分裂）'
    
    # 実例：ノズルからの水噴霧
    rho_water = 998.2  # kg/m³
    sigma_water = 0.0728  # N/m（20°C）
    
    # 噴霧速度とノズル径の範囲
    spray_velocity = np.linspace(1, 30, 50)  # m/s
    nozzle_diameter = 1e-3  # m (1 mm)
    
    We_values = weber_number(rho_water, spray_velocity, nozzle_diameter, sigma_water)
    
    print("=" * 70)
    print("Weber数と液滴分裂モード（水の噴霧）")
    print("=" * 70)
    print(f"流体: 水（20°C）、密度 = {rho_water} kg/m³、表面張力 = {sigma_water*1000:.2f} mN/m")
    print(f"ノズル径: {nozzle_diameter*1000:.2f} mm")
    print("-" * 70)
    
    test_velocities = [5, 10, 15, 20, 25]
    for v in test_velocities:
        We = weber_number(rho_water, v, nozzle_diameter, sigma_water)
        regime = droplet_regime(We)
        print(f"噴霧速度 {v:5.1f} m/s → We = {We:8.1f} → {regime}")
    
    # 可視化
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(spray_velocity, We_values, linewidth=3, color='#11998e',
            label='Weber数')
    
    # 分裂モードの境界線
    ax.axhline(y=1, color='green', linestyle='--', linewidth=2, label='変形開始 (We = 1)')
    ax.axhline(y=12, color='orange', linestyle='--', linewidth=2, label='バッグ分裂 (We = 12)')
    ax.axhline(y=100, color='red', linestyle='--', linewidth=2, label='微細化 (We = 100)')
    
    # 領域の塗りつぶし
    ax.fill_between(spray_velocity, 0, 1, alpha=0.2, color='green', label='安定領域')
    ax.fill_between(spray_velocity, 1, 12, alpha=0.2, color='yellow', label='変形領域')
    ax.fill_between(spray_velocity, 12, 100, alpha=0.2, color='orange', label='バッグ分裂領域')
    ax.fill_between(spray_velocity, 100, We_values.max(), alpha=0.2, color='red', label='微細化領域')
    
    ax.set_xlabel('噴霧速度 [m/s]', fontsize=12, fontweight='bold')
    ax.set_ylabel('Weber数 [-]', fontsize=12, fontweight='bold')
    ax.set_title('噴霧速度とWeber数：液滴分裂モード', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(alpha=0.3)
    ax.set_yscale('log')
    ax.set_ylim([0.1, 1000])
    
    plt.tight_layout()
    plt.show()
    

**出力:**
    
    
    ======================================================================
    Weber数と液滴分裂モード（水の噴霧）
    ======================================================================
    流体: 水（20°C）、密度 = 998.2 kg/m³、表面張力 = 72.8 mN/m
    ノズル径: 1.00 mm
    ----------------------------------------------------------------------
    噴霧速度   5.0 m/s → We =    342.5 → 微細化（カタストロフィック分裂）
    噴霧速度  10.0 m/s → We =   1370.1 → 微細化（カタストロフィック分裂）
    噴霧速度  15.0 m/s → We =   3082.7 → 微細化（カタストロフィック分裂）
    噴霧速度  20.0 m/s → We =   5480.5 → 微細化（カタストロフィック分裂）
    噴霧速度  25.0 m/s → We =   8563.5 → 微細化（カタストロフィック分裂）
    

**解説:** 工業用噴霧（5-30 m/s）では、Weber数が非常に高く、液滴は激しく分裂して微細化します。スケールアップ時に同じ液滴径分布を得るには、Weber数を一定に保つ必要があります。

* * *

## 2.3 攪拌・混合の無次元数

### Power数（動力数）

Power数（Po）は、攪拌に要する動力と慣性力の比を表します：

$$ \text{Po} = \frac{P}{\rho N^3 D^5} $$

ここで：

  * $P$: 攪拌動力 [W]
  * $N$: 回転数 [rps]
  * $D$: 攪拌翼径 [m]

**重要:** Power数はReynolds数の関数であり、$\text{Po} = f(\text{Re})$の関係があります。

### コード例4: Power数とReynolds数の関係（攪拌槽）
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    def power_number(Re, impeller_type='rushton'):
        """
        Reynolds数からPower数を推算（経験式）
    
        Parameters:
        -----------
        Re : float or array
            Reynolds数（攪拌槽）
        impeller_type : str
            攪拌翼の種類
            - 'rushton': Rushtonタービン（標準6枚翼）
            - 'paddle': パドル翼
            - 'propeller': プロペラ翼
    
        Returns:
        --------
        Po : float or array
            Power数
        """
        if impeller_type == 'rushton':
            # Rushtonタービンの経験式
            Po_turb = 5.0  # 乱流域
            Po_lam = 64 / Re  # 層流域
            Po = np.where(Re < 10, Po_lam,
                         np.where(Re < 10000,
                                 Po_turb * (1 + 10/Re),
                                 Po_turb))
        elif impeller_type == 'paddle':
            Po_turb = 1.5
            Po_lam = 50 / Re
            Po = np.where(Re < 10, Po_lam,
                         np.where(Re < 10000,
                                 Po_turb * (1 + 15/Re),
                                 Po_turb))
        elif impeller_type == 'propeller':
            Po_turb = 0.32
            Po_lam = 40 / Re
            Po = np.where(Re < 10, Po_lam,
                         np.where(Re < 5000,
                                 Po_turb * (1 + 10/Re),
                                 Po_turb))
        else:
            raise ValueError(f"Unknown impeller type: {impeller_type}")
    
        return Po
    
    def calculate_power(rho, N, D, mu, impeller_type='rushton'):
        """
        攪拌動力を計算
    
        Parameters:
        -----------
        rho : float
            流体密度 [kg/m³]
        N : float
            回転数 [rps]
        D : float
            攪拌翼径 [m]
        mu : float
            粘度 [Pa·s]
        impeller_type : str
            攪拌翼の種類
    
        Returns:
        --------
        P : float
            攪拌動力 [W]
        Re : float
            Reynolds数
        Po : float
            Power数
        """
        Re = rho * N * D**2 / mu
        Po = power_number(Re, impeller_type)
        P = Po * rho * N**3 * D**5
        return P, Re, Po
    
    # Reynolds数の範囲
    Re_range = np.logspace(0, 6, 500)
    
    # 各種攪拌翼のPower数
    Po_rushton = power_number(Re_range, 'rushton')
    Po_paddle = power_number(Re_range, 'paddle')
    Po_propeller = power_number(Re_range, 'propeller')
    
    # 可視化
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.loglog(Re_range, Po_rushton, linewidth=2.5, color='#11998e',
              label='Rushtonタービン')
    ax.loglog(Re_range, Po_paddle, linewidth=2.5, color='#e74c3c',
              label='パドル翼')
    ax.loglog(Re_range, Po_propeller, linewidth=2.5, color='#f39c12',
              label='プロペラ翼')
    
    ax.axvline(x=10, color='gray', linestyle='--', linewidth=1.5, alpha=0.5)
    ax.axvline(x=10000, color='gray', linestyle='--', linewidth=1.5, alpha=0.5)
    ax.text(5, 0.1, '層流域', fontsize=11, ha='right', color='gray')
    ax.text(100, 0.1, '遷移域', fontsize=11, ha='center', color='gray')
    ax.text(50000, 0.1, '乱流域', fontsize=11, ha='left', color='gray')
    
    ax.set_xlabel('Reynolds数 [-]', fontsize=12, fontweight='bold')
    ax.set_ylabel('Power数 [-]', fontsize=12, fontweight='bold')
    ax.set_title('Power数とReynolds数の関係（攪拌翼の種類別）',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(which='both', alpha=0.3)
    ax.set_xlim([1, 1e6])
    ax.set_ylim([0.05, 100])
    
    plt.tight_layout()
    plt.show()
    
    # 実例計算：水の攪拌（Rushtonタービン）
    print("=" * 70)
    print("攪拌動力計算の実例（水、Rushtonタービン）")
    print("=" * 70)
    
    rho_water = 998.2  # kg/m³
    mu_water = 1.002e-3  # Pa·s
    D = 0.2  # m (20 cm 翼径)
    N_values = np.array([1, 2, 3, 4, 5])  # rps
    
    for N in N_values:
        P, Re, Po = calculate_power(rho_water, N, D, mu_water, 'rushton')
        print(f"回転数 {N} rps → Re = {Re:10,.0f}, Po = {Po:.3f}, 動力 = {P:.2f} W")
    

**出力:**
    
    
    ======================================================================
    攪拌動力計算の実例（水、Rushtonタービン）
    ======================================================================
    回転数 1 rps → Re =     39,848, Po = 5.000, 動力 = 1.60 W
    回転数 2 rps → Re =     79,696, Po = 5.000, 動力 = 12.78 W
    回転数 3 rps → Re =    119,544, Po = 5.000, 動力 = 43.15 W
    回転数 4 rps → Re =    159,392, Po = 5.000, 動力 = 102.27 W
    回転数 5 rps → Re =    199,240, Po = 5.000, 動力 = 199.75 W
    

**解説:** 乱流域では、Power数は一定（Rushtonタービンでは約5）となります。動力は回転数の3乗、翼径の5乗に比例するため、スケールアップ時には急激に増加します。

* * *

## 2.4 伝熱・物質移動の無次元数（概要）

伝熱・物質移動のスケーリングでは、以下の無次元数が重要です（詳細は第3章で扱います）：

無次元数 | 定義式 | 物理的意味  
---|---|---  
**Nusselt数（Nu）** | $\text{Nu} = \frac{hL}{k}$ | 対流伝熱 / 熱伝導  
**Prandtl数（Pr）** | $\text{Pr} = \frac{\nu}{\alpha} = \frac{c_p \mu}{k}$ | 運動量拡散 / 熱拡散  
**Grashof数（Gr）** | $\text{Gr} = \frac{g \beta \Delta T L^3}{\nu^2}$ | 浮力 / 粘性力  
**Sherwood数（Sh）** | $\text{Sh} = \frac{k_c L}{D_{AB}}$ | 対流物質移動 / 拡散  
**Schmidt数（Sc）** | $\text{Sc} = \frac{\nu}{D_{AB}}$ | 運動量拡散 / 物質拡散  
  
* * *

## 2.5 多基準相似解析

### 複数の無次元数を同時に満たす問題

実際のスケーリングでは、複数の無次元数を同時に一致させることが求められます。しかし、すべての無次元数を同時に満たすことは**物理的に不可能** な場合があります。

**例:** 攪拌槽のスケールアップで、Reynolds数とFroude数を同時に一定に保つ場合：

  * Re一定 → $N \propto S^{-2}$
  * Fr一定 → $N \propto S^{-0.5}$

これらは矛盾するため、**どちらを優先するかの判断** が必要です。

### コード例5: 多基準相似解析（攪拌槽）
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    def scaling_analysis(S, criterion):
        """
        異なるスケーリング基準での回転数と動力の変化
    
        Parameters:
        -----------
        S : array
            スケール倍率
        criterion : str
            スケーリング基準
            - 'Re': Reynolds数一定
            - 'Fr': Froude数一定
            - 'P/V': 単位体積あたり動力一定
            - 'tip_speed': 翼端速度一定
    
        Returns:
        --------
        N_ratio : array
            回転数比（ラボスケール = 1）
        P_ratio : array
            動力比（ラボスケール = 1）
        """
        if criterion == 'Re':
            N_ratio = S**(-2)
            P_ratio = S**(-3)
        elif criterion == 'Fr':
            N_ratio = S**(-0.5)
            P_ratio = S**(2.5)
        elif criterion == 'P/V':
            N_ratio = S**(-1)
            P_ratio = S**(2)
        elif criterion == 'tip_speed':
            N_ratio = S**(-1)
            P_ratio = S**(2)
        else:
            raise ValueError(f"Unknown criterion: {criterion}")
    
        return N_ratio, P_ratio
    
    # スケール倍率
    S = np.array([1, 2, 5, 10, 20])
    
    # 各基準での計算
    criteria = ['Re', 'Fr', 'P/V', 'tip_speed']
    colors = ['#11998e', '#e74c3c', '#f39c12', '#9b59b6']
    
    print("=" * 80)
    print("多基準相似解析：スケールアップ時の回転数と動力の変化")
    print("=" * 80)
    print(f"{'スケール':>8} | {'Re一定':>15} | {'Fr一定':>15} | {'P/V一定':>15} | {'翼端速度一定':>15}")
    print(f"{'倍率':>8} | {'N比':>7} {'P比':>7} | {'N比':>7} {'P比':>7} | {'N比':>7} {'P比':>7} | {'N比':>7} {'P比':>7}")
    print("-" * 80)
    
    for s in S:
        N_Re, P_Re = scaling_analysis(np.array([s]), 'Re')
        N_Fr, P_Fr = scaling_analysis(np.array([s]), 'Fr')
        N_PV, P_PV = scaling_analysis(np.array([s]), 'P/V')
        N_tip, P_tip = scaling_analysis(np.array([s]), 'tip_speed')
    
        print(f"{s:8.0f} | {N_Re[0]:7.3f} {P_Re[0]:7.2f} | "
              f"{N_Fr[0]:7.3f} {P_Fr[0]:7.2f} | "
              f"{N_PV[0]:7.3f} {P_PV[0]:7.2f} | "
              f"{N_tip[0]:7.3f} {P_tip[0]:7.2f}")
    
    # 可視化
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    S_plot = np.linspace(1, 20, 100)
    
    for i, crit in enumerate(criteria):
        N_ratio, P_ratio = scaling_analysis(S_plot, crit)
    
        ax1.plot(S_plot, N_ratio, linewidth=2.5, color=colors[i],
                 label=f'{crit}一定')
        ax2.plot(S_plot, P_ratio, linewidth=2.5, color=colors[i],
                 label=f'{crit}一定')
    
    ax1.set_xlabel('スケール倍率 [-]', fontsize=12, fontweight='bold')
    ax1.set_ylabel('回転数比（N/N₀）[-]', fontsize=12, fontweight='bold')
    ax1.set_title('スケーリング基準別：回転数の変化', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(alpha=0.3)
    ax1.set_yscale('log')
    
    ax2.set_xlabel('スケール倍率 [-]', fontsize=12, fontweight='bold')
    ax2.set_ylabel('動力比（P/P₀）[-]', fontsize=12, fontweight='bold')
    ax2.set_title('スケーリング基準別：動力の変化', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(alpha=0.3)
    ax2.set_yscale('log')
    
    plt.tight_layout()
    plt.show()
    

**出力:**
    
    
    ================================================================================
    多基準相似解析：スケールアップ時の回転数と動力の変化
    ================================================================================
    スケール |         Re一定 |         Fr一定 |        P/V一定 |     翼端速度一定
        倍率 |    N比    P比 |    N比    P比 |    N比    P比 |    N比    P比
    --------------------------------------------------------------------------------
           1 |   1.000    1.00 |   1.000    1.00 |   1.000    1.00 |   1.000    1.00
           2 |   0.250    0.12 |   0.707    5.66 |   0.500    4.00 |   0.500    4.00
           5 |   0.040    0.01 |   0.447   56.57 |   0.200   25.00 |   0.200   25.00
          10 |   0.010    0.00 |   0.316  316.23 |   0.100  100.00 |   0.100  100.00
          20 |   0.003    0.00 |   0.224 1788.85 |   0.050  400.00 |   0.050  400.00
    

**解説:**

  * **Reynolds数一定:** 回転数・動力ともに急激に減少。高粘度流体でのスケーリングに適用。
  * **Froude数一定:** 動力が大幅に増加。自由表面流・ボルテックス制御が重要な系に適用。
  * **P/V一定:** 単位体積あたりの動力を保つ。一般的な混合・懸濁に適用。
  * **翼端速度一定:** せん断速度を保つ。せん断感受性物質（細胞培養等）に適用。

* * *

### コード例6: 支配的な無次元数の判定
    
    
    import numpy as np
    
    def identify_dominant_forces(Re, Fr, We):
        """
        Reynolds数、Froude数、Weber数から支配的な力を判定
    
        Parameters:
        -----------
        Re : float
            Reynolds数
        Fr : float
            Froude数
        We : float
            Weber数
    
        Returns:
        --------
        dominant_forces : dict
            支配的な力のリスト
        """
        forces = []
    
        # Reynolds数の評価
        if Re < 1:
            forces.append('粘性力（層流、Re < 1）')
        elif Re < 2300:
            forces.append('粘性力（層流、Re < 2,300）')
        else:
            forces.append('慣性力（乱流、Re > 2,300）')
    
        # Froude数の評価
        if Fr < 0.5:
            forces.append('重力（Fr < 0.5、重力支配）')
        elif Fr > 2:
            forces.append('慣性力（Fr > 2、慣性支配）')
        else:
            forces.append('重力・慣性バランス（0.5 < Fr < 2）')
    
        # Weber数の評価
        if We < 1:
            forces.append('表面張力（We < 1、表面張力支配）')
        elif We > 10:
            forces.append('慣性力（We > 10、液滴分裂）')
        else:
            forces.append('表面張力・慣性バランス（1 < We < 10）')
    
        return forces
    
    # 実例：各種化学プロセスの無次元数解析
    processes = [
        {
            'name': '攪拌槽（水、低速）',
            'Re': 5000,
            'Fr': 0.3,
            'We': 50
        },
        {
            'name': '攪拌槽（水、高速）',
            'Re': 100000,
            'Fr': 2.5,
            'We': 500
        },
        {
            'name': '高粘度流体の攪拌',
            'Re': 10,
            'Fr': 0.05,
            'We': 0.5
        },
        {
            'name': 'ノズル噴霧',
            'Re': 50000,
            'Fr': 15,
            'We': 5000
        },
        {
            'name': '気泡塔',
            'Re': 1000,
            'Fr': 1.0,
            'We': 5
        }
    ]
    
    print("=" * 80)
    print("各種プロセスにおける支配的な力の解析")
    print("=" * 80)
    
    for proc in processes:
        print(f"\n【{proc['name']}】")
        print(f"  Reynolds数 = {proc['Re']:,.0f}")
        print(f"  Froude数   = {proc['Fr']:.2f}")
        print(f"  Weber数    = {proc['We']:.0f}")
        print(f"  支配的な力:")
    
        dominant = identify_dominant_forces(proc['Re'], proc['Fr'], proc['We'])
        for i, force in enumerate(dominant, 1):
            print(f"    {i}. {force}")
    

**出力:**
    
    
    ================================================================================
    各種プロセスにおける支配的な力の解析
    ================================================================================
    
    【攪拌槽（水、低速）】
      Reynolds数 = 5,000
      Froude数   = 0.30
      Weber数    = 50
      支配的な力:
        1. 慣性力（乱流、Re > 2,300）
        2. 重力（Fr < 0.5、重力支配）
        3. 慣性力（We > 10、液滴分裂）
    
    【攪拌槽（水、高速）】
      Reynolds数 = 100,000
      Froude数   = 2.50
      Weber数    = 500
      支配的な力:
        1. 慣性力（乱流、Re > 2,300）
        2. 慣性力（Fr > 2、慣性支配）
        3. 慣性力（We > 10、液滴分裂）
    
    【高粘度流体の攪拌】
      Reynolds数 = 10
      Froude数   = 0.05
      Weber数    = 0.5
      支配的な力:
        1. 粘性力（層流、Re < 2,300）
        2. 重力（Fr < 0.5、重力支配）
        3. 表面張力（We < 1、表面張力支配）
    
    【ノズル噴霧】
      Reynolds数 = 50,000
      Froude数   = 15.00
      Weber数    = 5000
      支配的な力:
        1. 慣性力（乱流、Re > 2,300）
        2. 慣性力（Fr > 2、慣性支配）
        3. 慣性力（We > 10、液滴分裂）
    
    【気泡塔】
      Reynolds数 = 1,000
      Froude数   = 1.00
      Weber数    = 5
      支配的な力:
        1. 粘性力（層流、Re < 2,300）
        2. 重力・慣性バランス（0.5 < Fr < 2）
        3. 表面張力・慣性バランス（1 < We < 10）
    

**解説:** 各プロセスで支配的な力が異なります。スケーリングでは、支配的な力に対応する無次元数を優先的に一致させることが重要です。

* * *

### コード例7: スケーリング基準選択のデシジョンツリー
    
    
    def recommend_scaling_criterion(process_type, fluid_viscosity, has_free_surface,
                                     shear_sensitive, phase='single'):
        """
        プロセス特性に基づいて推奨スケーリング基準を提案
    
        Parameters:
        -----------
        process_type : str
            プロセスタイプ（'mixing', 'reaction', 'separation', 'dispersion'）
        fluid_viscosity : str
            流体粘度（'low': < 100 mPa·s, 'medium': 100-1000, 'high': > 1000）
        has_free_surface : bool
            自由表面の有無
        shear_sensitive : bool
            せん断感受性の有無
        phase : str
            相状態（'single', 'gas-liquid', 'liquid-liquid', 'solid-liquid'）
    
        Returns:
        --------
        recommendations : dict
            推奨基準とその理由
        """
        recommendations = {
            'primary': None,
            'secondary': None,
            'reason': ''
        }
    
        # 高粘度流体の場合
        if fluid_viscosity == 'high':
            recommendations['primary'] = 'Reynolds数一定'
            recommendations['reason'] = '高粘度流体では層流域となるため、Re一定で粘性流動を保つ'
            recommendations['secondary'] = '翼端速度一定（せん断速度制御）'
    
        # せん断感受性物質
        elif shear_sensitive:
            recommendations['primary'] = '翼端速度一定'
            recommendations['reason'] = 'せん断感受性物質の損傷を防ぐため、せん断速度を一定に保つ'
            recommendations['secondary'] = 'P/V一定（エネルギー散逸率制御）'
    
        # 自由表面がある場合
        elif has_free_surface:
            recommendations['primary'] = 'Froude数一定'
            recommendations['reason'] = '自由表面のボルテックス形成を相似に保つ'
            recommendations['secondary'] = 'P/V一定'
    
        # 気液分散
        elif phase == 'gas-liquid':
            recommendations['primary'] = 'P/V一定'
            recommendations['reason'] = '気泡径とガスホールドアップを相似に保つ'
            recommendations['secondary'] = 'Weber数一定（気泡分裂制御）'
    
        # 液液分散
        elif phase == 'liquid-liquid':
            recommendations['primary'] = 'Weber数一定'
            recommendations['reason'] = '液滴径分布を相似に保つ'
            recommendations['secondary'] = 'P/V一定'
    
        # 一般的な混合
        else:
            recommendations['primary'] = 'P/V一定'
            recommendations['reason'] = '混合時間と乱流エネルギー散逸率を相似に保つ'
            recommendations['secondary'] = 'Reynolds数一定（流動パターン制御）'
    
        return recommendations
    
    # 実例
    test_cases = [
        {
            'name': '水の一般攪拌',
            'process_type': 'mixing',
            'fluid_viscosity': 'low',
            'has_free_surface': False,
            'shear_sensitive': False,
            'phase': 'single'
        },
        {
            'name': '高粘度ポリマー溶液',
            'process_type': 'mixing',
            'fluid_viscosity': 'high',
            'has_free_surface': False,
            'shear_sensitive': True,
            'phase': 'single'
        },
        {
            'name': '気液反応器（バブルカラム）',
            'process_type': 'reaction',
            'fluid_viscosity': 'low',
            'has_free_surface': True,
            'shear_sensitive': False,
            'phase': 'gas-liquid'
        },
        {
            'name': '乳化プロセス',
            'process_type': 'dispersion',
            'fluid_viscosity': 'medium',
            'has_free_surface': False,
            'shear_sensitive': False,
            'phase': 'liquid-liquid'
        },
        {
            'name': '細胞培養',
            'process_type': 'mixing',
            'fluid_viscosity': 'low',
            'has_free_surface': False,
            'shear_sensitive': True,
            'phase': 'single'
        }
    ]
    
    print("=" * 80)
    print("プロセス特性に基づくスケーリング基準の推奨")
    print("=" * 80)
    
    for case in test_cases:
        rec = recommend_scaling_criterion(
            case['process_type'],
            case['fluid_viscosity'],
            case['has_free_surface'],
            case['shear_sensitive'],
            case['phase']
        )
    
        print(f"\n【{case['name']}】")
        print(f"  プロセスタイプ: {case['process_type']}")
        print(f"  粘度: {case['fluid_viscosity']}, 自由表面: {case['has_free_surface']}, "
              f"せん断感受性: {case['shear_sensitive']}, 相: {case['phase']}")
        print(f"  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print(f"  推奨スケーリング基準:")
        print(f"    第1優先: {rec['primary']}")
        print(f"    第2優先: {rec['secondary']}")
        print(f"  理由: {rec['reason']}")
    

**出力:**
    
    
    ================================================================================
    プロセス特性に基づくスケーリング基準の推奨
    ================================================================================
    
    【水の一般攪拌】
      プロセスタイプ: mixing
      粘度: low, 自由表面: False, せん断感受性: False, 相: single
      ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
      推奨スケーリング基準:
        第1優先: P/V一定
        第2優先: Reynolds数一定（流動パターン制御）
      理由: 混合時間と乱流エネルギー散逸率を相似に保つ
    
    【高粘度ポリマー溶液】
      プロセスタイプ: mixing
      粘度: high, 自由表面: False, せん断感受性: True, 相: single
      ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
      推奨スケーリング基準:
        第1優先: Reynolds数一定
        第2優先: 翼端速度一定（せん断速度制御）
      理由: 高粘度流体では層流域となるため、Re一定で粘性流動を保つ
    
    【気液反応器（バブルカラム）】
      プロセスタイプ: reaction
      粘度: low, 自由表面: True, せん断感受性: False, 相: gas-liquid
      ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
      推奨スケーリング基準:
        第1優先: Froude数一定
        第2優先: P/V一定
      理由: 自由表面のボルテックス形成を相似に保つ
    
    【乳化プロセス】
      プロセスタイプ: dispersion
      粘度: medium, 自由表面: False, せん断感受性: False, 相: liquid-liquid
      ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
      推奨スケーリング基準:
        第1優先: Weber数一定
        第2優先: P/V一定
      理由: 液滴径分布を相似に保つ
    
    【細胞培養】
      プロセスタイプ: mixing
      粘度: low, 自由表面: False, せん断感受性: True, 相: single
      ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
      推奨スケーリング基準:
        第1優先: 翼端速度一定
        第2優先: P/V一定（エネルギー散逸率制御）
      理由: せん断感受性物質の損傷を防ぐため、せん断速度を一定に保つ
    

**解説:** プロセス特性に応じて最適なスケーリング基準が異なります。このデシジョンツリーは、実務でのスケーリング戦略立案の出発点となります。

* * *

## 2.6 相似則の限界

### 完全相似が不可能な理由

すべての無次元数を同時に一致させることは、以下の理由から不可能です：

  1. **独立な無次元数が多すぎる:** 流体力学だけでもRe, Fr, We, Ca（Capillary数）等が存在
  2. **物性値の制約:** 密度、粘度、表面張力は独立に変更できない
  3. **幾何学的制約:** 壁面効果、アスペクト比の限界
  4. **実用的制約:** 動力、圧力損失、建設コスト

### 部分相似（Partial Similarity）

実際のスケーリングでは、**支配的な現象に対応する無次元数を優先的に一致させ、他は許容範囲内の変動を認める** という「部分相似」のアプローチを取ります。

プロセス | 優先する無次元数 | 許容する無次元数  
---|---|---  
高速攪拌（低粘度） | P/V一定、Re確保（乱流） | Froude数は変化を許容  
自由表面攪拌 | Froude数一定 | Reynolds数は変化を許容  
気液分散 | P/V一定、Weber数 | Reynolds数は乱流域確保のみ  
高粘度流体 | Reynolds数一定 | Froude数、Weber数は無視  
  
* * *

## 学習目標の確認

この章を完了すると、以下を説明できるようになります：

### 基本理解

  * ✅ 幾何学的、運動学的、動力学的相似の3階層を説明できる
  * ✅ 主要な無次元数（Re, Fr, We, Po）の物理的意味を理解している
  * ✅ 無次元数と力のバランスの関係を説明できる

### 実践スキル

  * ✅ Pythonで各種無次元数を計算し、流動状態を判定できる
  * ✅ スケーリング基準（Re一定、Fr一定、P/V一定等）の違いを定量化できる
  * ✅ プロセス特性から適切なスケーリング基準を選択できる

### 応用力

  * ✅ 複数の無次元数を用いた多基準相似解析を実行できる
  * ✅ 完全相似が不可能な理由と部分相似の必要性を理解している
  * ✅ 実プロセスで支配的な力を同定し、スケーリング戦略を立案できる

* * *

### 免責事項

  * 本資料の内容は教育目的で作成されており、実際のプロセス設計には専門家の助言が必要です
  * 数値例は典型的な条件での概算値であり、実際の系では物性値・操作条件の検証が必須です
  * スケールアップには安全性評価、リスクアセスメント、段階的検証が不可欠です
  * 無次元数の相関式は経験式であり、適用範囲外では精度が低下します
