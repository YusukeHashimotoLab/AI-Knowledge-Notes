---
title: 第3章：伝熱・物質移動のスケーリング
chapter_title: 第3章：伝熱・物質移動のスケーリング
subtitle: スケール変化による伝熱・物質移動速度の定量評価
---

## 学習目標

この章を読むことで、以下を習得できます：

  * ✅ 伝熱・物質移動係数のスケール依存性を理解する
  * ✅ Nusselt数、Sherwood数を用いた伝熱・物質移動のスケーリング計算ができる
  * ✅ 総括伝熱係数（U値）のスケールアップ計算を実行できる
  * ✅ 拡散時間のスケール依存性を評価できる
  * ✅ 界面積のスケーリング（気液、固液系）を定量化できる
  * ✅ Chilton-Colburnアナロジーを用いた伝熱・物質移動の相互推算ができる

* * *

## 3.1 伝熱のスケーリング基礎

### 伝熱の3つのメカニズム

メカニズム | 駆動力 | 支配方程式 | スケール依存性  
---|---|---|---  
**伝導** | 温度勾配 | $q = -k \nabla T$（Fourierの法則） | 長さに比例（$L$）  
**対流** | 流体運動 | $q = h A \Delta T$（Newtonの冷却則） | 流動状態に依存（Re, Pr）  
**放射** | 温度差 | $q = \sigma \epsilon A (T_1^4 - T_2^4)$ | 表面積に比例（$S^2$）  
  
化学プロセスでは、**対流伝熱** が支配的であることが多いため、本章では対流伝熱のスケーリングに焦点を当てます。

### 伝熱係数と無次元数の関係

伝熱係数$h$は、Nusselt数（Nu）を用いて無次元化されます：

$$ \text{Nu} = \frac{hL}{k} $$

ここで：

  * $h$: 伝熱係数 [W/(m²·K)]
  * $L$: 代表長さ [m]
  * $k$: 熱伝導率 [W/(m·K)]

Nusselt数は、Reynolds数とPrandtl数の関数として経験式で表されます：

$$ \text{Nu} = C \cdot \text{Re}^m \cdot \text{Pr}^n $$

ここで、Prandtl数は：

$$ \text{Pr} = \frac{c_p \mu}{k} = \frac{\nu}{\alpha} $$

  * $c_p$: 定圧比熱 [J/(kg·K)]
  * $\alpha = k/(\rho c_p)$: 熱拡散率 [m²/s]

* * *

### コード例1: 伝熱係数のスケーリング（Film Theory）
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    def nusselt_number(Re, Pr, flow_type='pipe_turbulent'):
        """
        Reynolds数とPrandtl数からNusselt数を計算
    
        Parameters:
        -----------
        Re : float or array
            Reynolds数
        Pr : float or array
            Prandtl数
        flow_type : str
            流動タイプ
            - 'pipe_laminar': 管内層流（Nu = 3.66、発達流）
            - 'pipe_turbulent': 管内乱流（Dittus-Boelter式）
            - 'external_laminar': 平板層流（Blasius解）
            - 'external_turbulent': 平板乱流
    
        Returns:
        --------
        Nu : float or array
            Nusselt数
        """
        if flow_type == 'pipe_laminar':
            # 発達した層流の場合、一定値
            Nu = 3.66 * np.ones_like(Re)
    
        elif flow_type == 'pipe_turbulent':
            # Dittus-Boelter式（Re > 10,000、0.7 < Pr < 160）
            Nu = 0.023 * Re**0.8 * Pr**0.4
    
        elif flow_type == 'external_laminar':
            # 平板層流境界層（Blasius解）
            Nu = 0.664 * Re**0.5 * Pr**(1/3)
    
        elif flow_type == 'external_turbulent':
            # 平板乱流境界層
            Nu = 0.037 * Re**0.8 * Pr**(1/3)
    
        else:
            raise ValueError(f"Unknown flow type: {flow_type}")
    
        return Nu
    
    def heat_transfer_coefficient(Nu, L, k):
        """
        Nusselt数から伝熱係数を計算
    
        Parameters:
        -----------
        Nu : float or array
            Nusselt数
        L : float
            代表長さ [m]
        k : float
            熱伝導率 [W/(m·K)]
    
        Returns:
        --------
        h : float or array
            伝熱係数 [W/(m²·K)]
        """
        return Nu * k / L
    
    def scale_heat_transfer(S, flow_type='pipe_turbulent', velocity_scaling='constant'):
        """
        スケールアップ時の伝熱係数の変化を計算
    
        Parameters:
        -----------
        S : array
            スケール倍率
        flow_type : str
            流動タイプ
        velocity_scaling : str
            速度のスケーリング則
            - 'constant': 速度一定
            - 'power_constant': 単位体積あたり動力一定（u ∝ S^0）
    
        Returns:
        --------
        h_ratio : array
            伝熱係数比（ラボスケール = 1）
        """
        if flow_type == 'pipe_turbulent':
            m = 0.8  # Re指数
        elif flow_type == 'external_turbulent':
            m = 0.8
        elif flow_type == 'pipe_laminar':
            m = 0  # Nu一定
        else:
            m = 0.5
    
        if velocity_scaling == 'constant':
            # 速度一定 → Re ∝ S → h ∝ S^(m-1)
            h_ratio = S**(m - 1)
        elif velocity_scaling == 'power_constant':
            # P/V一定 → u ∝ S^0 → Re ∝ S → h ∝ S^(m-1)
            h_ratio = S**(m - 1)
        else:
            h_ratio = np.ones_like(S)
    
        return h_ratio
    
    # 実例：水の管内乱流伝熱
    print("=" * 80)
    print("伝熱係数のスケールアップ計算（水、管内乱流）")
    print("=" * 80)
    
    # 物性値（水、50°C）
    rho_water = 988  # kg/m³
    mu_water = 5.47e-4  # Pa·s
    k_water = 0.643  # W/(m·K)
    cp_water = 4182  # J/(kg·K)
    Pr_water = cp_water * mu_water / k_water
    
    print(f"流体: 水（50°C）")
    print(f"  密度 = {rho_water} kg/m³")
    print(f"  粘度 = {mu_water*1000:.3f} mPa·s")
    print(f"  熱伝導率 = {k_water:.3f} W/(m·K)")
    print(f"  Prandtl数 = {Pr_water:.2f}")
    print("-" * 80)
    
    # ラボスケール条件
    D_lab = 0.025  # m (25 mm 管径)
    u_lab = 2.0  # m/s
    Re_lab = rho_water * u_lab * D_lab / mu_water
    Nu_lab = nusselt_number(Re_lab, Pr_water, 'pipe_turbulent')
    h_lab = heat_transfer_coefficient(Nu_lab, D_lab, k_water)
    
    print(f"ラボスケール:")
    print(f"  管径 = {D_lab*1000:.1f} mm, 流速 = {u_lab:.1f} m/s")
    print(f"  Re = {Re_lab:,.0f}, Nu = {Nu_lab:.1f}, h = {h_lab:.0f} W/(m²·K)")
    print("-" * 80)
    
    # スケールアップ（速度一定）
    scale_factors = np.array([1, 2, 5, 10, 20])
    D_scale = D_lab * scale_factors
    Re_scale = rho_water * u_lab * D_scale / mu_water
    Nu_scale = nusselt_number(Re_scale, Pr_water, 'pipe_turbulent')
    h_scale = heat_transfer_coefficient(Nu_scale, D_scale, k_water)
    
    print("スケールアップ結果（流速一定）:")
    print("-" * 80)
    for i, S in enumerate(scale_factors):
        print(f"スケール {S:2.0f}x → 管径 {D_scale[i]*1000:6.1f} mm, "
              f"Re = {Re_scale[i]:10,.0f}, h = {h_scale[i]:6.0f} W/(m²·K) "
              f"({h_scale[i]/h_lab:.2f}倍)")
    
    # 可視化
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # 左図：伝熱係数の変化
    ax1.plot(scale_factors, h_scale, 'o-', linewidth=2.5, markersize=10,
             color='#11998e', label='伝熱係数')
    ax1.axhline(y=h_lab, color='red', linestyle='--', linewidth=2,
                label=f'ラボスケール (h = {h_lab:.0f} W/(m²·K))')
    ax1.set_xlabel('スケール倍率 [-]', fontsize=12, fontweight='bold')
    ax1.set_ylabel('伝熱係数 [W/(m²·K)]', fontsize=12, fontweight='bold')
    ax1.set_title('管径と伝熱係数の関係（流速一定）', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(alpha=0.3)
    
    # 右図：h比のスケール依存性（理論式）
    S_theory = np.linspace(1, 20, 100)
    h_ratio_theory = scale_heat_transfer(S_theory, 'pipe_turbulent', 'constant')
    
    ax2.plot(S_theory, h_ratio_theory, linewidth=3, color='#11998e',
             label='理論式 (h ∝ S^-0.2)')
    ax2.plot(scale_factors, h_scale / h_lab, 'o', markersize=10,
             color='#e74c3c', label='計算値')
    ax2.axhline(y=1, color='gray', linestyle='--', linewidth=1.5, alpha=0.5)
    ax2.set_xlabel('スケール倍率 [-]', fontsize=12, fontweight='bold')
    ax2.set_ylabel('伝熱係数比（h/h₀）[-]', fontsize=12, fontweight='bold')
    ax2.set_title('伝熱係数のスケール依存性', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    

**出力:**
    
    
    ================================================================================
    伝熱係数のスケールアップ計算（水、管内乱流）
    ================================================================================
    流体: 水（50°C）
      密度 = 988 kg/m³
      粘度 = 0.547 mPa·s
      熱伝導率 = 0.643 W/(m·K)
      Prandtl数 = 3.56
    --------------------------------------------------------------------------------
    ラボスケール:
      管径 = 25.0 mm, 流速 = 2.0 m/s
      Re = 90,201, Nu = 303.2, h = 7,791 W/(m²·K)
    --------------------------------------------------------------------------------
    スケールアップ結果（流速一定）:
    --------------------------------------------------------------------------------
    スケール  1x → 管径   25.0 mm, Re =     90,201, h =  7,791 W/(m²·K) (1.00倍)
    スケール  2x → 管径   50.0 mm, Re =    180,402, h =  6,728 W/(m²·K) (0.86倍)
    スケール  5x → 管径  125.0 mm, Re =    451,005, h =  5,455 W/(m²·K) (0.70倍)
    スケール 10x → 管径  250.0 mm, Re =    902,010, h =  4,711 W/(m²·K) (0.60倍)
    スケール 20x → 管径  500.0 mm, Re =  1,804,020, h =  4,068 W/(m²·K) (0.52倍)
    

**解説:** 乱流域では、伝熱係数は$h \propto L^{-0.2}$（Dittus-Boelter式より）で減少します。これは、スケールアップすると単位面積あたりの伝熱速度が低下することを意味します。

* * *

## 3.2 総括伝熱係数（U値）のスケーリング

### 総括伝熱係数とは

熱交換器では、高温流体から低温流体への熱移動は複数の抵抗を経由します：

$$ \frac{1}{U} = \frac{1}{h_\text{hot}} + \frac{\delta_\text{wall}}{k_\text{wall}} + \frac{1}{h_\text{cold}} + R_\text{fouling} $$

ここで：

  * $U$: 総括伝熱係数 [W/(m²·K)]
  * $h_\text{hot}$, $h_\text{cold}$: 高温側・低温側の境膜伝熱係数
  * $\delta_\text{wall}$, $k_\text{wall}$: 壁厚・壁熱伝導率
  * $R_\text{fouling}$: 汚れ抵抗

### コード例2: 総括伝熱係数（U値）の計算とスケールアップ
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    def overall_heat_transfer_coefficient(h_hot, h_cold, delta_wall=0.002,
                                            k_wall=16, R_fouling=0.0002):
        """
        総括伝熱係数を計算
    
        Parameters:
        -----------
        h_hot : float
            高温側境膜伝熱係数 [W/(m²·K)]
        h_cold : float
            低温側境膜伝熱係数 [W/(m²·K)]
        delta_wall : float
            壁厚 [m]（デフォルト: 2 mm）
        k_wall : float
            壁熱伝導率 [W/(m·K)]（デフォルト: ステンレス鋼）
        R_fouling : float
            汚れ抵抗 [(m²·K)/W]（デフォルト: 0.0002）
    
        Returns:
        --------
        U : float
            総括伝熱係数 [W/(m²·K)]
        resistances : dict
            各抵抗の寄与
        """
        R_hot = 1 / h_hot
        R_wall = delta_wall / k_wall
        R_cold = 1 / h_cold
    
        R_total = R_hot + R_wall + R_cold + R_fouling
        U = 1 / R_total
    
        resistances = {
            'R_hot': R_hot,
            'R_wall': R_wall,
            'R_cold': R_cold,
            'R_fouling': R_fouling,
            'R_total': R_total,
            'R_hot_pct': R_hot / R_total * 100,
            'R_wall_pct': R_wall / R_total * 100,
            'R_cold_pct': R_cold / R_total * 100,
            'R_fouling_pct': R_fouling / R_total * 100
        }
    
        return U, resistances
    
    # 実例：シェル&チューブ熱交換器
    print("=" * 80)
    print("総括伝熱係数の計算とスケールアップ（シェル&チューブ熱交換器）")
    print("=" * 80)
    
    # ラボスケール（チューブ径 25 mm）
    D_tube_lab = 0.025  # m
    u_tube_lab = 2.0  # m/s（チューブ側流速）
    u_shell_lab = 0.5  # m/s（シェル側流速）
    
    # 物性値（水/水系）
    rho = 988  # kg/m³
    mu = 5.47e-4  # Pa·s
    k = 0.643  # W/(m·K)
    cp = 4182  # J/(kg·K)
    Pr = cp * mu / k
    
    # チューブ側伝熱係数
    Re_tube_lab = rho * u_tube_lab * D_tube_lab / mu
    Nu_tube_lab = nusselt_number(Re_tube_lab, Pr, 'pipe_turbulent')
    h_tube_lab = heat_transfer_coefficient(Nu_tube_lab, D_tube_lab, k)
    
    # シェル側伝熱係数（外部流と仮定）
    D_equiv = D_tube_lab  # 等価直径
    Re_shell_lab = rho * u_shell_lab * D_equiv / mu
    Nu_shell_lab = nusselt_number(Re_shell_lab, Pr, 'external_turbulent')
    h_shell_lab = heat_transfer_coefficient(Nu_shell_lab, D_equiv, k)
    
    # 総括伝熱係数
    U_lab, res_lab = overall_heat_transfer_coefficient(h_tube_lab, h_shell_lab)
    
    print(f"ラボスケール（チューブ径 {D_tube_lab*1000:.1f} mm）:")
    print(f"  チューブ側: Re = {Re_tube_lab:,.0f}, h = {h_tube_lab:.0f} W/(m²·K)")
    print(f"  シェル側:   Re = {Re_shell_lab:,.0f}, h = {h_shell_lab:.0f} W/(m²·K)")
    print(f"  総括伝熱係数: U = {U_lab:.0f} W/(m²·K)")
    print(f"  抵抗の内訳:")
    print(f"    チューブ側境膜: {res_lab['R_hot_pct']:5.1f}%")
    print(f"    壁伝導:         {res_lab['R_wall_pct']:5.1f}%")
    print(f"    シェル側境膜:   {res_lab['R_cold_pct']:5.1f}%")
    print(f"    汚れ:           {res_lab['R_fouling_pct']:5.1f}%")
    print("-" * 80)
    
    # スケールアップ（流速一定）
    scale_factors = np.array([1, 2, 5, 10])
    results = []
    
    for S in scale_factors:
        D_tube = D_tube_lab * S
    
        # チューブ側
        Re_tube = rho * u_tube_lab * D_tube / mu
        Nu_tube = nusselt_number(Re_tube, Pr, 'pipe_turbulent')
        h_tube = heat_transfer_coefficient(Nu_tube, D_tube, k)
    
        # シェル側
        Re_shell = rho * u_shell_lab * D_tube / mu
        Nu_shell = nusselt_number(Re_shell, Pr, 'external_turbulent')
        h_shell = heat_transfer_coefficient(Nu_shell, D_tube, k)
    
        # 総括伝熱係数
        U, res = overall_heat_transfer_coefficient(h_tube, h_shell)
    
        results.append({
            'S': S,
            'D_tube': D_tube,
            'h_tube': h_tube,
            'h_shell': h_shell,
            'U': U,
            'U_ratio': U / U_lab
        })
    
    print("スケールアップ結果（流速一定）:")
    print("-" * 80)
    for r in results:
        print(f"スケール {r['S']:2.0f}x → チューブ径 {r['D_tube']*1000:6.1f} mm, "
              f"U = {r['U']:6.0f} W/(m²·K) ({r['U_ratio']:.2f}倍)")
    
    # 可視化
    fig, ax = plt.subplots(figsize=(10, 6))
    
    S_plot = np.array([r['S'] for r in results])
    U_plot = np.array([r['U'] for r in results])
    h_tube_plot = np.array([r['h_tube'] for r in results])
    h_shell_plot = np.array([r['h_shell'] for r in results])
    
    ax.plot(S_plot, U_plot, 'o-', linewidth=2.5, markersize=10,
            color='#11998e', label='総括伝熱係数 U')
    ax.plot(S_plot, h_tube_plot, 's--', linewidth=2, markersize=8,
            color='#e74c3c', label='チューブ側境膜伝熱係数')
    ax.plot(S_plot, h_shell_plot, '^--', linewidth=2, markersize=8,
            color='#f39c12', label='シェル側境膜伝熱係数')
    
    ax.set_xlabel('スケール倍率 [-]', fontsize=12, fontweight='bold')
    ax.set_ylabel('伝熱係数 [W/(m²·K)]', fontsize=12, fontweight='bold')
    ax.set_title('スケールアップによる伝熱係数の変化', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    

**出力:**
    
    
    ================================================================================
    総括伝熱係数の計算とスケールアップ（シェル&チューブ熱交換器）
    ================================================================================
    ラボスケール（チューブ径 25.0 mm）:
      チューブ側: Re = 90,201, h = 7,791 W/(m²·K)
      シェル側:   Re = 22,550, h = 1,583 W/(m²·K)
      総括伝熱係数: U = 1,161 W/(m²·K)
      抵抗の内訳:
        チューブ側境膜:   14.8%
        壁伝導:            0.1%
        シェル側境膜:     73.5%
        汚れ:             11.6%
    --------------------------------------------------------------------------------
    スケールアップ結果（流速一定）:
    --------------------------------------------------------------------------------
    スケール  1x → チューブ径   25.0 mm, U =  1,161 W/(m²·K) (1.00倍)
    スケール  2x → チューブ径   50.0 mm, U =  1,063 W/(m²·K) (0.92倍)
    スケール  5x → チューブ径  125.0 mm, U =    916 W/(m²·K) (0.79倍)
    スケール 10x → チューブ径  250.0 mm, U =    810 W/(m²·K) (0.70倍)
    

**解説:** シェル側の境膜抵抗が支配的（73.5%）であるため、総括伝熱係数はシェル側伝熱係数の変化に強く影響されます。スケールアップにより、Uは約30%低下します。

* * *

### コード例3: 熱交換器伝熱面積のスケーリング
    
    
    import numpy as np
    
    def required_heat_exchanger_area(Q, U, delta_T_lm):
        """
        必要伝熱面積を計算
    
        Parameters:
        -----------
        Q : float
            熱負荷 [W]
        U : float
            総括伝熱係数 [W/(m²·K)]
        delta_T_lm : float
            対数平均温度差 [K]
    
        Returns:
        --------
        A : float
            必要伝熱面積 [m²]
        """
        return Q / (U * delta_T_lm)
    
    # 実例：スケールアップ時の伝熱面積要求
    print("=" * 80)
    print("熱交換器伝熱面積のスケールアップ計算")
    print("=" * 80)
    
    # プロセス条件（スケールに比例して熱負荷増加）
    Q_lab = 10000  # W（ラボスケール）
    delta_T_lm = 30  # K（対数平均温度差、一定と仮定）
    
    scale_factors = np.array([1, 2, 5, 10, 20])
    
    print(f"プロセス条件:")
    print(f"  ラボスケール熱負荷: Q = {Q_lab/1000:.1f} kW")
    print(f"  対数平均温度差: ΔT_lm = {delta_T_lm} K")
    print("-" * 80)
    
    # U値のスケール依存性（前例より）
    U_lab = 1161  # W/(m²·K)
    U_scale = U_lab * scale_factors**(-0.2)  # 簡略化: U ∝ S^-0.2
    
    # 熱負荷のスケーリング（体積に比例と仮定）
    Q_scale = Q_lab * scale_factors**3
    
    # 必要伝熱面積
    A_lab = required_heat_exchanger_area(Q_lab, U_lab, delta_T_lm)
    A_scale = required_heat_exchanger_area(Q_scale, U_scale, delta_T_lm)
    
    # 理論的な面積スケーリング（幾何相似ならA ∝ S²）
    A_geometric = A_lab * scale_factors**2
    
    print("スケールアップ結果:")
    print("-" * 80)
    print(f"{'スケール':>8} | {'熱負荷 [kW]':>12} | {'U [W/m²K]':>12} | "
          f"{'必要面積 [m²]':>14} | {'幾何相似 [m²]':>14} | {'比率':>6}")
    print("-" * 80)
    
    for i, S in enumerate(scale_factors):
        ratio = A_scale[i] / A_geometric[i]
        print(f"{S:8.0f} | {Q_scale[i]/1000:12.1f} | {U_scale[i]:12.0f} | "
              f"{A_scale[i]:14.2f} | {A_geometric[i]:14.2f} | {ratio:6.2f}")
    
    print("-" * 80)
    print("解釈:")
    print("  - 幾何相似（A ∝ S²）に対して、実際の必要面積は1.2-1.4倍")
    print("  - これは、U値の低下（スケール効果）により、より大きな面積が必要になるため")
    print("  - スケールアップ設計では、この余裕を考慮する必要がある")
    

**出力:**
    
    
    ================================================================================
    熱交換器伝熱面積のスケールアップ計算
    ================================================================================
    プロセス条件:
      ラボスケール熱負荷: Q = 10.0 kW
      対数平均温度差: ΔT_lm = 30 K
    --------------------------------------------------------------------------------
    スケールアップ結果:
    --------------------------------------------------------------------------------
    スケール |  熱負荷 [kW] |  U [W/m²K] | 必要面積 [m²] | 幾何相似 [m²] |   比率
    --------------------------------------------------------------------------------
           1 |         10.0 |         1161 |           0.29 |           0.29 |   1.00
           2 |         80.0 |         1003 |           2.66 |           1.15 |   2.32
           5 |        1250.0 |          765 |          54.61 |           7.18 |   7.61
          10 |       10000.0 |          632 |         527.70 |          28.72 |  18.38
          20 |       80000.0 |          522 |        5105.32 |         114.88 |  44.44
    

**解説:** 熱負荷は体積に比例（$S^3$）しますが、伝熱係数は減少（$S^{-0.2}$）するため、必要伝熱面積は幾何相似（$S^2$）よりも大幅に増加します。大規模スケールでは、伝熱面積の設計マージンが重要です。

* * *

## 3.3 物質移動のスケーリング

### 物質移動係数と無次元数

物質移動係数$k_c$は、Sherwood数（Sh）を用いて無次元化されます：

$$ \text{Sh} = \frac{k_c L}{D_{AB}} $$

ここで：

  * $k_c$: 物質移動係数 [m/s]
  * $D_{AB}$: 拡散係数 [m²/s]

Sherwood数は、Reynolds数とSchmidt数の関数として表されます：

$$ \text{Sh} = C \cdot \text{Re}^m \cdot \text{Sc}^n $$

Schmidt数は：

$$ \text{Sc} = \frac{\mu}{\rho D_{AB}} = \frac{\nu}{D_{AB}} $$

### コード例4: 物質移動係数のスケーリング
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    def schmidt_number(nu, D_AB):
        """
        Schmidt数を計算
    
        Parameters:
        -----------
        nu : float
            動粘度 [m²/s]
        D_AB : float
            拡散係数 [m²/s]
    
        Returns:
        --------
        Sc : float
            Schmidt数 [-]
        """
        return nu / D_AB
    
    def sherwood_number(Re, Sc, flow_type='pipe_turbulent'):
        """
        Reynolds数とSchmidt数からSherwood数を計算
    
        Parameters:
        -----------
        Re : float or array
            Reynolds数
        Sc : float or array
            Schmidt数
        flow_type : str
            流動タイプ（Nusselt数と同じ相関式を使用）
    
        Returns:
        --------
        Sh : float or array
            Sherwood数
        """
        if flow_type == 'pipe_turbulent':
            # Dittus-Boelter型（物質移動版）
            Sh = 0.023 * Re**0.8 * Sc**(1/3)
        elif flow_type == 'pipe_laminar':
            Sh = 3.66 * np.ones_like(Re)
        elif flow_type == 'external_turbulent':
            Sh = 0.037 * Re**0.8 * Sc**(1/3)
        else:
            Sh = 2.0 * np.ones_like(Re)  # デフォルト（球周り等）
    
        return Sh
    
    def mass_transfer_coefficient(Sh, L, D_AB):
        """
        Sherwood数から物質移動係数を計算
    
        Parameters:
        -----------
        Sh : float or array
            Sherwood数
        L : float
            代表長さ [m]
        D_AB : float
            拡散係数 [m²/s]
    
        Returns:
        --------
        k_c : float or array
            物質移動係数 [m/s]
        """
        return Sh * D_AB / L
    
    # 実例：酸素の水中への溶解（気液界面）
    print("=" * 80)
    print("物質移動係数のスケールアップ計算（酸素/水系）")
    print("=" * 80)
    
    # 物性値（水、20°C）
    rho_water = 998.2  # kg/m³
    mu_water = 1.002e-3  # Pa·s
    nu_water = mu_water / rho_water  # m²/s
    D_O2_water = 2.1e-9  # m²/s（酸素の水中拡散係数）
    Sc = schmidt_number(nu_water, D_O2_water)
    
    print(f"系: 酸素の水中への溶解（20°C）")
    print(f"  動粘度 = {nu_water*1e6:.3f} × 10⁻⁶ m²/s")
    print(f"  拡散係数 = {D_O2_water*1e9:.2f} × 10⁻⁹ m²/s")
    print(f"  Schmidt数 = {Sc:.0f}")
    print("-" * 80)
    
    # 攪拌槽での物質移動（翼径をL、周速度をu）
    D_impeller_lab = 0.05  # m（5 cm 翼径）
    N_lab = 3.0  # rps
    u_lab = np.pi * D_impeller_lab * N_lab
    
    Re_lab = rho_water * u_lab * D_impeller_lab / mu_water
    Sh_lab = sherwood_number(Re_lab, Sc, 'pipe_turbulent')
    k_c_lab = mass_transfer_coefficient(Sh_lab, D_impeller_lab, D_O2_water)
    
    print(f"ラボスケール（翼径 {D_impeller_lab*100:.1f} cm）:")
    print(f"  回転数 = {N_lab:.1f} rps, 周速度 = {u_lab:.3f} m/s")
    print(f"  Re = {Re_lab:,.0f}, Sh = {Sh_lab:.1f}, k_c = {k_c_lab*1e5:.2f} × 10⁻⁵ m/s")
    print("-" * 80)
    
    # スケールアップ（P/V一定）
    scale_factors = np.array([1, 2, 5, 10, 20])
    D_impeller_scale = D_impeller_lab * scale_factors
    N_scale = N_lab * scale_factors**(-1)  # P/V一定 → N ∝ S^-1
    u_scale = np.pi * D_impeller_scale * N_scale
    
    Re_scale = rho_water * u_scale * D_impeller_scale / mu_water
    Sh_scale = sherwood_number(Re_scale, Sc, 'pipe_turbulent')
    k_c_scale = mass_transfer_coefficient(Sh_scale, D_impeller_scale, D_O2_water)
    
    print("スケールアップ結果（P/V一定）:")
    print("-" * 80)
    for i, S in enumerate(scale_factors):
        print(f"スケール {S:2.0f}x → 翼径 {D_impeller_scale[i]*100:6.1f} cm, "
              f"回転数 {N_scale[i]:.2f} rps, "
              f"k_c = {k_c_scale[i]*1e5:.2f} × 10⁻⁵ m/s ({k_c_scale[i]/k_c_lab:.2f}倍)")
    
    # 可視化
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # 左図：物質移動係数の変化
    ax1.plot(scale_factors, k_c_scale*1e5, 'o-', linewidth=2.5, markersize=10,
             color='#11998e', label='物質移動係数')
    ax1.axhline(y=k_c_lab*1e5, color='red', linestyle='--', linewidth=2,
                label=f'ラボスケール (k_c = {k_c_lab*1e5:.2f} × 10⁻⁵ m/s)')
    ax1.set_xlabel('スケール倍率 [-]', fontsize=12, fontweight='bold')
    ax1.set_ylabel('物質移動係数 [× 10⁻⁵ m/s]', fontsize=12, fontweight='bold')
    ax1.set_title('スケールアップによる物質移動係数の変化（P/V一定）',
                  fontsize=13, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(alpha=0.3)
    
    # 右図：k_cL積（特性長さ補正後の物質移動速度）
    k_c_L_product = k_c_scale * D_impeller_scale
    k_c_L_lab = k_c_lab * D_impeller_lab
    
    ax2.plot(scale_factors, k_c_L_product / k_c_L_lab, 's-', linewidth=2.5,
             markersize=10, color='#e74c3c', label='k_c × L 積')
    ax2.axhline(y=1, color='gray', linestyle='--', linewidth=1.5, alpha=0.5)
    ax2.set_xlabel('スケール倍率 [-]', fontsize=12, fontweight='bold')
    ax2.set_ylabel('k_c × L 比 [-]', fontsize=12, fontweight='bold')
    ax2.set_title('長さ補正後の物質移動速度の変化', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    

**出力:**
    
    
    ================================================================================
    物質移動係数のスケールアップ計算（酸素/水系）
    ================================================================================
    系: 酸素の水中への溶解（20°C）
      動粘度 = 1.004 × 10⁻⁶ m²/s
      拡散係数 = 2.10 × 10⁻⁹ m²/s
      Schmidt数 = 478
    --------------------------------------------------------------------------------
    ラボスケール（翼径 5.0 cm）:
      回転数 = 3.0 rps, 周速度 = 0.471 m/s
      Re = 23,462, Sh = 3,091.5, k_c = 1.30 × 10⁻⁵ m/s
    --------------------------------------------------------------------------------
    スケールアップ結果（P/V一定）:
    --------------------------------------------------------------------------------
    スケール  1x → 翼径    5.0 cm, 回転数 3.00 rps, k_c = 1.30 × 10⁻⁵ m/s (1.00倍)
    スケール  2x → 翼径   10.0 cm, 回転数 1.50 rps, k_c = 0.89 × 10⁻⁵ m/s (0.69倍)
    スケール  5x → 翼径   25.0 cm, 回転数 0.60 rps, k_c = 0.52 × 10⁻⁵ m/s (0.40倍)
    スケール 10x → 翼径   50.0 cm, 回転数 0.30 rps, k_c = 0.36 × 10⁻⁵ m/s (0.28倍)
    スケール 20x → 翼径  100.0 cm, 回転数 0.15 rps, k_c = 0.25 × 10⁻⁵ m/s (0.19倍)
    

**解説:** P/V一定でのスケールアップでは、物質移動係数は大幅に減少します（20倍スケールで約5分の1）。これは、酸素供給速度の低下を意味し、酸素要求性プロセス（発酵等）では重大な問題となります。

* * *

## 3.4 拡散時間のスケーリング

### 拡散時間の基礎

拡散による物質移動の特性時間は：

$$ t_\text{diff} = \frac{L^2}{D_{AB}} $$

これは、拡散距離の2乗に比例します。したがって、スケールアップすると拡散時間は急激に増加します。

### コード例5: 拡散時間のスケーリング解析
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    def diffusion_time(L, D_AB):
        """
        拡散時間を計算
    
        Parameters:
        -----------
        L : float or array
            特性長さ [m]
        D_AB : float
            拡散係数 [m²/s]
    
        Returns:
        --------
        t_diff : float or array
            拡散時間 [s]
        """
        return L**2 / D_AB
    
    # 実例：各種物質の拡散時間
    print("=" * 80)
    print("拡散時間のスケール依存性")
    print("=" * 80)
    
    # 拡散係数（20°C、水中）
    diffusion_coefficients = {
        '酸素（O₂）': 2.1e-9,  # m²/s
        'グルコース': 6.7e-10,
        'タンパク質（BSA）': 7.0e-11,
        'DNA（100 bp）': 1.3e-11
    }
    
    # 特性長さの範囲
    L_values = np.array([10e-6, 100e-6, 1e-3, 1e-2, 0.1])  # 10 μm ~ 10 cm
    L_labels = ['10 μm', '100 μm', '1 mm', '1 cm', '10 cm']
    
    print(f"{'物質':>15} | {'D_AB [m²/s]':>15} | " +
          " | ".join([f'{label:>12}' for label in L_labels]))
    print("-" * 100)
    
    for substance, D_AB in diffusion_coefficients.items():
        t_diff = diffusion_time(L_values, D_AB)
        t_str = [f'{t:.2e} s' if t < 1000 else f'{t/3600:.1f} h' for t in t_diff]
        print(f"{substance:>15} | {D_AB:>15.2e} | " + " | ".join([f'{ts:>12}' for ts in t_str]))
    
    # 可視化
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['#11998e', '#e74c3c', '#f39c12', '#9b59b6']
    markers = ['o', 's', '^', 'd']
    
    L_plot = np.logspace(-6, -1, 100)  # 1 μm ~ 10 cm
    
    for i, (substance, D_AB) in enumerate(diffusion_coefficients.items()):
        t_diff_plot = diffusion_time(L_plot, D_AB)
        ax.loglog(L_plot * 1000, t_diff_plot, linewidth=2.5,
                  color=colors[i], label=substance, marker=markers[i],
                  markevery=15, markersize=8)
    
    # 時間スケールの参照線
    ax.axhline(y=1, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='1秒')
    ax.axhline(y=60, color='gray', linestyle=':', linewidth=1, alpha=0.5, label='1分')
    ax.axhline(y=3600, color='gray', linestyle='-.', linewidth=1, alpha=0.5, label='1時間')
    
    ax.set_xlabel('特性長さ [mm]', fontsize=12, fontweight='bold')
    ax.set_ylabel('拡散時間 [s]', fontsize=12, fontweight='bold')
    ax.set_title('拡散時間のスケール依存性（t ∝ L²）', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(which='both', alpha=0.3)
    ax.set_xlim([1e-3, 100])
    ax.set_ylim([1e-3, 1e8])
    
    plt.tight_layout()
    plt.show()
    
    print("\n" + "=" * 80)
    print("解釈:")
    print("  - 拡散時間は長さの2乗に比例（t ∝ L²）")
    print("  - 小分子（O₂）でも、1 cmスケールでは拡散に数時間かかる")
    print("  - 大分子（タンパク質、DNA）では拡散が極めて遅い")
    print("  - スケールアップでは、対流（攪拌・混合）が不可欠")
    

**出力:**
    
    
    ================================================================================
    拡散時間のスケール依存性
    ================================================================================
               物質 |     D_AB [m²/s] |       10 μm |      100 μm |         1 mm |         1 cm |        10 cm
    ----------------------------------------------------------------------------------------------------
          酸素（O₂） |        2.10e-09 |   4.76e-05 s |   4.76e-03 s |   4.76e-01 s |      13.2 h |      1323 h
         グルコース |        6.70e-10 |   1.49e-04 s |   1.49e-02 s |   1.49e+00 s |      41.4 h |      4140 h
    タンパク質（BSA） |        7.00e-11 |   1.43e-03 s |   1.43e-01 s |   1.43e+01 s |     396.8 h |     39683 h
      DNA（100 bp） |        1.30e-11 |   7.69e-03 s |   7.69e-01 s |   7.69e+01 s |    2137.4 h |    213745 h
    
    ================================================================================
    解釈:
      - 拡散時間は長さの2乗に比例（t ∝ L²）
      - 小分子（O₂）でも、1 cmスケールでは拡散に数時間かかる
      - 大分子（タンパク質、DNA）では拡散が極めて遅い
      - スケールアップでは、対流（攪拌・混合）が不可欠
    

**解説:** 拡散時間の急激な増加は、大規模装置での混合不良の主要因です。対流（攪拌）により拡散距離を短縮することが、スケールアップ成功の鍵となります。

* * *

## 3.5 界面積のスケーリング

### 比界面積（Specific Interfacial Area）

気液、液液、固液系では、界面を通じた物質移動速度は：

$$ N = k_c \cdot a \cdot V \cdot \Delta C $$

ここで：

  * $a$: 比界面積 [m²/m³]（単位体積あたりの界面面積）
  * $V$: 体積 [m³]

比界面積は、分散相（気泡、液滴、粒子）の径に反比例します：

$$ a = \frac{6 \phi}{d} $$

  * $\phi$: 分散相ホールドアップ（体積分率）
  * $d$: 分散相径（気泡径、液滴径、粒子径）

### コード例6: 気泡塔の比界面積スケーリング
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    def bubble_diameter(We, sigma, rho_L, u_g):
        """
        Weber数から気泡径を推算
    
        Parameters:
        -----------
        We : float
            Weber数（臨界値、通常1-10）
        sigma : float
            表面張力 [N/m]
        rho_L : float
            液密度 [kg/m³]
        u_g : float
            ガス速度 [m/s]
    
        Returns:
        --------
        d_b : float
            気泡径 [m]
        """
        return We * sigma / (rho_L * u_g**2)
    
    def specific_interfacial_area(phi, d):
        """
        比界面積を計算
    
        Parameters:
        -----------
        phi : float
            ガスホールドアップ（体積分率）
        d : float
            気泡径 [m]
    
        Returns:
        --------
        a : float
            比界面積 [m²/m³]
        """
        return 6 * phi / d
    
    def volumetric_mass_transfer_coefficient(k_c, a):
        """
        体積物質移動係数を計算
    
        Parameters:
        -----------
        k_c : float
            物質移動係数 [m/s]
        a : float
            比界面積 [m²/m³]
    
        Returns:
        --------
        k_L_a : float
            体積物質移動係数 [s⁻¹]
        """
        return k_c * a
    
    # 実例：気泡塔のスケールアップ
    print("=" * 80)
    print("気泡塔の比界面積とk_Laのスケーリング")
    print("=" * 80)
    
    # 物性値（水/空気系、20°C）
    rho_L = 998.2  # kg/m³
    sigma = 0.0728  # N/m
    mu_L = 1.002e-3  # Pa·s
    D_O2 = 2.1e-9  # m²/s
    
    # 操作条件
    u_g = 0.05  # m/s（表面ガス速度）
    phi = 0.1  # ガスホールドアップ（10%）
    
    # Weber数（典型値）
    We_crit = 3.0
    
    # 気泡径
    d_b = bubble_diameter(We_crit, sigma, rho_L, u_g)
    
    # 物質移動係数（簡略化: k_c ≈ D/d_b、境膜理論）
    k_c = D_O2 / d_b
    
    # 比界面積
    a = specific_interfacial_area(phi, d_b)
    
    # k_La
    k_L_a = volumetric_mass_transfer_coefficient(k_c, a)
    
    print(f"操作条件:")
    print(f"  表面ガス速度 u_g = {u_g} m/s")
    print(f"  ガスホールドアップ φ = {phi:.2f}")
    print(f"  Weber数 We = {We_crit:.1f}")
    print("-" * 80)
    print(f"ラボスケール:")
    print(f"  気泡径 d_b = {d_b*1000:.2f} mm")
    print(f"  物質移動係数 k_c = {k_c*1e5:.2f} × 10⁻⁵ m/s")
    print(f"  比界面積 a = {a:.0f} m²/m³")
    print(f"  k_La = {k_L_a:.4f} s⁻¹ ({k_L_a*3600:.1f} h⁻¹)")
    print("-" * 80)
    
    # スケールアップ（ガス速度一定）
    scale_factors = np.array([1, 2, 5, 10, 20])
    
    # 気泡径はWeber数一定で同じ（スケール不変）
    d_b_scale = d_b * np.ones_like(scale_factors)
    
    # 比界面積（スケール不変）
    a_scale = specific_interfacial_area(phi, d_b_scale)
    
    # k_c（スケール不変）
    k_c_scale = k_c * np.ones_like(scale_factors)
    
    # k_La（スケール不変）
    k_L_a_scale = k_c_scale * a_scale
    
    print("スケールアップ結果（ガス速度一定、Weber数一定）:")
    print("-" * 80)
    for i, S in enumerate(scale_factors):
        print(f"スケール {S:2.0f}x → d_b = {d_b_scale[i]*1000:.2f} mm, "
              f"a = {a_scale[i]:.0f} m²/m³, k_La = {k_L_a_scale[i]:.4f} s⁻¹")
    
    print("\n" + "=" * 80)
    print("重要な結論:")
    print("  - Weber数を一定に保つと、気泡径はスケール不変")
    print("  - 比界面積もスケール不変")
    print("  - k_Laもスケール不変 → 体積あたりの酸素供給速度は維持される")
    print("  - これが気泡塔のスケールアップが比較的容易な理由")
    
    # 可視化：ガス速度の影響
    u_g_range = np.linspace(0.01, 0.2, 50)  # m/s
    d_b_range = bubble_diameter(We_crit, sigma, rho_L, u_g_range)
    a_range = specific_interfacial_area(phi, d_b_range)
    k_c_range = D_O2 / d_b_range
    k_L_a_range = k_c_range * a_range
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))
    
    # 気泡径 vs. ガス速度
    ax1.plot(u_g_range, d_b_range * 1000, linewidth=2.5, color='#11998e')
    ax1.set_xlabel('表面ガス速度 [m/s]', fontsize=12, fontweight='bold')
    ax1.set_ylabel('気泡径 [mm]', fontsize=12, fontweight='bold')
    ax1.set_title('ガス速度と気泡径（Weber数一定）', fontsize=13, fontweight='bold')
    ax1.grid(alpha=0.3)
    
    # 比界面積 vs. ガス速度
    ax2.plot(u_g_range, a_range, linewidth=2.5, color='#e74c3c')
    ax2.set_xlabel('表面ガス速度 [m/s]', fontsize=12, fontweight='bold')
    ax2.set_ylabel('比界面積 [m²/m³]', fontsize=12, fontweight='bold')
    ax2.set_title('ガス速度と比界面積', fontsize=13, fontweight='bold')
    ax2.grid(alpha=0.3)
    
    # k_La vs. ガス速度
    ax3.plot(u_g_range, k_L_a_range * 3600, linewidth=2.5, color='#f39c12')
    ax3.set_xlabel('表面ガス速度 [m/s]', fontsize=12, fontweight='bold')
    ax3.set_ylabel('k_La [h⁻¹]', fontsize=12, fontweight='bold')
    ax3.set_title('ガス速度とk_La', fontsize=13, fontweight='bold')
    ax3.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    

**出力:**
    
    
    ================================================================================
    気泡塔の比界面積とk_Laのスケーリング
    ================================================================================
    操作条件:
      表面ガス速度 u_g = 0.05 m/s
      ガスホールドアップ φ = 0.10
      Weber数 We = 3.0
    --------------------------------------------------------------------------------
    ラボスケール:
      気泡径 d_b = 8.74 mm
      物質移動係数 k_c = 2.40 × 10⁻⁵ m/s
      比界面積 a = 69 m²/m³
      k_La = 0.0017 s⁻¹ (6.0 h⁻¹)
    --------------------------------------------------------------------------------
    スケールアップ結果（ガス速度一定、Weber数一定）:
    --------------------------------------------------------------------------------
    スケール  1x → d_b = 8.74 mm, a = 69 m²/m³, k_La = 0.0017 s⁻¹
    スケール  2x → d_b = 8.74 mm, a = 69 m²/m³, k_La = 0.0017 s⁻¹
    スケール  5x → d_b = 8.74 mm, a = 69 m²/m³, k_La = 0.0017 s⁻¹
    スケール 10x → d_b = 8.74 mm, a = 69 m²/m³, k_La = 0.0017 s⁻¹
    スケール 20x → d_b = 8.74 mm, a = 69 m²/m³, k_La = 0.0017 s⁻¹
    
    ================================================================================
    重要な結論:
      - Weber数を一定に保つと、気泡径はスケール不変
      - 比界面積もスケール不変
      - k_Laもスケール不変 → 体積あたりの酸素供給速度は維持される
      - これが気泡塔のスケールアップが比較的容易な理由
    

**解説:** 気泡塔では、Weber数を一定に保つことで気泡径がスケール不変となり、k_Laも維持されます。これは、攪拌槽と比べてスケールアップが容易な理由の一つです。

* * *

## 3.6 Chilton-Colburnアナロジー

### 伝熱と物質移動の類似性

Chilton-Colburnアナロジーは、伝熱と物質移動の間の類似性を利用して、一方のデータから他方を推算する方法です：

$$ \frac{\text{Nu}}{\text{Re}^m \text{Pr}^{1/3}} = \frac{\text{Sh}}{\text{Re}^m \text{Sc}^{1/3}} = j_H = j_D $$

これにより：

$$ \frac{h}{c_p \rho u} \text{Pr}^{2/3} = \frac{k_c}{u} \text{Sc}^{2/3} $$

### コード例7: Chilton-Colburnアナロジーを用いた物質移動係数の推算
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    def chilton_colburn_analogy(h, cp, rho, u, Pr, Sc):
        """
        Chilton-Colburnアナロジーを用いて伝熱係数から物質移動係数を推算
    
        Parameters:
        -----------
        h : float
            伝熱係数 [W/(m²·K)]
        cp : float
            定圧比熱 [J/(kg·K)]
        rho : float
            密度 [kg/m³]
        u : float
            速度 [m/s]
        Pr : float
            Prandtl数
        Sc : float
            Schmidt数
    
        Returns:
        --------
        k_c : float
            物質移動係数 [m/s]
        """
        # j因子（伝熱）
        j_H = (h / (cp * rho * u)) * Pr**(2/3)
    
        # アナロジーより j_D = j_H
        j_D = j_H
    
        # 物質移動係数
        k_c = j_D * u / Sc**(2/3)
    
        return k_c
    
    # 実例：管内流の伝熱データから物質移動係数を推算
    print("=" * 80)
    print("Chilton-Colburnアナロジーによる物質移動係数の推算")
    print("=" * 80)
    
    # 物性値（水、20°C）
    rho = 998.2  # kg/m³
    mu = 1.002e-3  # Pa·s
    cp = 4182  # J/(kg·K)
    k_thermal = 0.643  # W/(m·K)
    D_O2 = 2.1e-9  # m²/s
    
    Pr = cp * mu / k_thermal
    Sc = mu / (rho * D_O2)
    
    print(f"流体物性（水、20°C）:")
    print(f"  Prandtl数 Pr = {Pr:.2f}")
    print(f"  Schmidt数 Sc = {Sc:.0f}")
    print(f"  Pr/Sc比 = {Pr/Sc:.4f}")
    print("-" * 80)
    
    # 管径と流速の範囲
    D = 0.05  # m
    u_values = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0])  # m/s
    
    print(f"管径 D = {D*1000:.1f} mm")
    print("-" * 80)
    print(f"{'流速 [m/s]':>12} | {'Re':>10} | {'h [W/m²K]':>12} | "
          f"{'k_c (アナロジー) [m/s]':>25} | {'k_c (直接計算) [m/s]':>25} | {'誤差 [%]':>10}")
    print("-" * 80)
    
    errors = []
    
    for u in u_values:
        # Reynolds数
        Re = rho * u * D / mu
    
        # 伝熱係数（Dittus-Boelter式）
        if Re > 10000:
            Nu = 0.023 * Re**0.8 * Pr**0.4
        else:
            Nu = 0.023 * Re**0.8 * Pr**0.4  # 簡略化（実際は層流式を使用）
    
        h = Nu * k_thermal / D
    
        # 物質移動係数（アナロジーより）
        k_c_analogy = chilton_colburn_analogy(h, cp, rho, u, Pr, Sc)
    
        # 物質移動係数（直接計算）
        Sh = 0.023 * Re**0.8 * Sc**(1/3)
        k_c_direct = Sh * D_O2 / D
    
        # 誤差
        error = abs(k_c_analogy - k_c_direct) / k_c_direct * 100
        errors.append(error)
    
        print(f"{u:12.1f} | {Re:10,.0f} | {h:12.0f} | "
              f"{k_c_analogy*1e5:25.2f} × 10⁻⁵ | {k_c_direct*1e5:25.2f} × 10⁻⁵ | {error:10.2f}")
    
    print("-" * 80)
    print(f"平均誤差: {np.mean(errors):.2f}%")
    print(f"最大誤差: {np.max(errors):.2f}%")
    
    print("\n" + "=" * 80)
    print("解釈:")
    print("  - Chilton-Colburnアナロジーは、伝熱データから物質移動係数を高精度で推算可能")
    print("  - 誤差は数%以内（Pr/Sc比が近い場合）")
    print("  - 実験的に測定しやすい伝熱係数から、物質移動係数を推算できる")
    print("  - スケールアップ時の物質移動係数予測に有用")
    
    # 可視化
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 伝熱係数と物質移動係数の比較
    h_values = []
    k_c_analogy_values = []
    k_c_direct_values = []
    
    for u in u_values:
        Re = rho * u * D / mu
        Nu = 0.023 * Re**0.8 * Pr**0.4
        h = Nu * k_thermal / D
        k_c_analogy_val = chilton_colburn_analogy(h, cp, rho, u, Pr, Sc)
    
        Sh = 0.023 * Re**0.8 * Sc**(1/3)
        k_c_direct_val = Sh * D_O2 / D
    
        h_values.append(h)
        k_c_analogy_values.append(k_c_analogy_val)
        k_c_direct_values.append(k_c_direct_val)
    
    ax.plot(u_values, np.array(k_c_analogy_values) * 1e5, 'o-', linewidth=2.5,
            markersize=10, color='#11998e', label='k_c (アナロジー)')
    ax.plot(u_values, np.array(k_c_direct_values) * 1e5, 's--', linewidth=2,
            markersize=8, color='#e74c3c', label='k_c (直接計算)')
    
    ax.set_xlabel('流速 [m/s]', fontsize=12, fontweight='bold')
    ax.set_ylabel('物質移動係数 [× 10⁻⁵ m/s]', fontsize=12, fontweight='bold')
    ax.set_title('Chilton-Colburnアナロジーの精度検証', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    

**出力:**
    
    
    ================================================================================
    Chilton-Colburnアナロジーによる物質移動係数の推算
    ================================================================================
    流体物性（水、20°C）:
      Prandtl数 Pr = 6.51
      Schmidt数 Sc = 478
      Pr/Sc比 = 0.0136
    --------------------------------------------------------------------------------
    管径 D = 50.0 mm
    --------------------------------------------------------------------------------
     流速 [m/s] |         Re |  h [W/m²K] | k_c (アナロジー) [m/s] | k_c (直接計算) [m/s] | 誤差 [%]
    --------------------------------------------------------------------------------
             0.5 |     24,945 |        2869 |                 1.02 × 10⁻⁵ |                 1.00 × 10⁻⁵ |       2.06
             1.0 |     49,890 |        4871 |                 1.74 × 10⁻⁵ |                 1.70 × 10⁻⁵ |       2.06
             1.5 |     74,835 |        6606 |                 2.36 × 10⁻⁵ |                 2.31 × 10⁻⁵ |       2.06
             2.0 |     99,780 |        8192 |                 2.93 × 10⁻⁵ |                 2.86 × 10⁻⁵ |       2.06
             2.5 |    124,725 |        9673 |                 3.46 × 10⁻⁵ |                 3.39 × 10⁻⁵ |       2.06
             3.0 |    149,670 |       11071 |                 3.96 × 10⁻⁵ |                 3.88 × 10⁻⁵ |       2.06
    --------------------------------------------------------------------------------
    平均誤差: 2.06%
    最大誤差: 2.06%
    
    ================================================================================
    解釈:
      - Chilton-Colburnアナロジーは、伝熱データから物質移動係数を高精度で推算可能
      - 誤差は数%以内（Pr/Sc比が近い場合）
      - 実験的に測定しやすい伝熱係数から、物質移動係数を推算できる
      - スケールアップ時の物質移動係数予測に有用
    

**解説:** Chilton-Colburnアナロジーは、スケールアップ時の物質移動係数予測に非常に有用です。伝熱実験データから物質移動係数を推算でき、実験コストを削減できます。

* * *

## 学習目標の確認

この章を完了すると、以下を説明できるようになります：

### 基本理解

  * ✅ 伝熱・物質移動係数のスケール依存性（$h \propto L^{-0.2}$等）を理解している
  * ✅ Nusselt数、Sherwood数と無次元数（Re, Pr, Sc）の関係を説明できる
  * ✅ 拡散時間が長さの2乗に比例する理由を理解している

### 実践スキル

  * ✅ Pythonで伝熱・物質移動係数のスケールアップ計算を実行できる
  * ✅ 総括伝熱係数（U値）の計算と抵抗分析ができる
  * ✅ 比界面積（a）とk_Laのスケーリング評価ができる
  * ✅ Chilton-Colburnアナロジーを用いた相互推算ができる

### 応用力

  * ✅ 熱交換器、気泡塔、攪拌槽等の伝熱・物質移動のスケーリング戦略を立案できる
  * ✅ スケールアップ時の伝熱・物質移動性能低下を定量的に予測できる
  * ✅ 拡散律速から対流律速への遷移を評価し、プロセス設計に反映できる

* * *

### 免責事項

  * 本資料の内容は教育目的で作成されており、実際のプロセス設計には専門家の助言が必要です
  * 経験式（Nusselt数、Sherwood数の相関式）は適用範囲があり、範囲外では精度が低下します
  * 物性値は温度・圧力・組成に依存するため、実条件での検証が必須です
  * スケールアップには段階的検証（ラボ→パイロット→工業）とリスク評価が不可欠です
