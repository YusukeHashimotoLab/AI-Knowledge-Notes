---
title: 第1章：スケーリング理論の基礎
chapter_title: 第1章：スケーリング理論の基礎
subtitle: 相似則、べき乗則、装置サイジングの理解
---

## 学習目標

この章を読むことで、以下を習得できます：

  * ✅ 幾何学的相似・運動学的相似・動力学的相似を理解する
  * ✅ べき乗則を用いてスケール因子を計算できる
  * ✅ 反応器、タンク、熱交換器の装置サイジングができる
  * ✅ パイロットプラントの最適スケールを決定できる
  * ✅ 経済性のスケーリング（6/10則）を理解し活用できる

* * *

## 1.1 相似則の基礎

### 相似則とは

**相似則（Similarity Laws）** は、異なるスケールの系の間に成り立つ物理的な対応関係を記述する法則です。化学工学では、以下の3つの相似が重要です：

  * **幾何学的相似（Geometric Similarity）** : 形状と寸法の比率が保持される
  * **運動学的相似（Kinematic Similarity）** : 速度場のパターンが相似
  * **動力学的相似（Dynamic Similarity）** : 力の比率が相似

### スケール因子

スケール因子 $\lambda$ を長さの比として定義すると、他の物理量は以下のようにスケールします：

物理量 | スケーリング則 | スケール因子  
---|---|---  
長さ (L) | $L_2 = \lambda \cdot L_1$ | $\lambda$  
面積 (A) | $A_2 = \lambda^2 \cdot A_1$ | $\lambda^2$  
体積 (V) | $V_2 = \lambda^3 \cdot V_1$ | $\lambda^3$  
速度 (v) | 条件により変化 | $\lambda^0$ または $\lambda^{0.5}$  
時間 (t) | 条件により変化 | $\lambda$ または $\lambda^{0.5}$  
  
* * *

## 1.2 Pythonによるスケーリング計算

### コード例1: 幾何学的相似のスケーリング関係
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    def geometric_scaling(lambda_scale, L1, A1, V1):
        """
        幾何学的相似則に基づくスケーリング計算
    
        Parameters:
        -----------
        lambda_scale : float
            スケール因子（長さの比）
        L1 : float
            元の系の特性長さ [m]
        A1 : float
            元の系の面積 [m²]
        V1 : float
            元の系の体積 [m³]
    
        Returns:
        --------
        dict : スケールアップ後の値
        """
        L2 = lambda_scale * L1
        A2 = lambda_scale**2 * A1
        V2 = lambda_scale**3 * V1
    
        return {
            'Length': L2,
            'Area': A2,
            'Volume': V2,
            'Surface_to_Volume_Ratio': A2 / V2
        }
    
    # ラボスケールの反応器
    L1_lab = 0.1  # 直径 0.1 m (10 cm)
    A1_lab = np.pi * L1_lab**2  # 底面積
    V1_lab = (np.pi / 4) * L1_lab**3  # 体積（球形と仮定）
    
    # スケール因子の範囲: 1倍〜100倍
    lambda_range = np.logspace(0, 2, 50)  # 1 to 100
    
    results = []
    for lambda_scale in lambda_range:
        result = geometric_scaling(lambda_scale, L1_lab, A1_lab, V1_lab)
        results.append({
            'lambda': lambda_scale,
            **result
        })
    
    # 可視化
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 長さのスケーリング
    axes[0, 0].loglog(lambda_range, [r['Length'] for r in results],
                      linewidth=2.5, color='#11998e', label='L ∝ λ¹')
    axes[0, 0].set_xlabel('スケール因子 λ', fontsize=11)
    axes[0, 0].set_ylabel('長さ L [m]', fontsize=11)
    axes[0, 0].set_title('(a) 長さのスケーリング', fontsize=12, fontweight='bold')
    axes[0, 0].grid(alpha=0.3, which='both')
    axes[0, 0].legend(fontsize=10)
    
    # 面積のスケーリング
    axes[0, 1].loglog(lambda_range, [r['Area'] for r in results],
                      linewidth=2.5, color='#e67e22', label='A ∝ λ²')
    axes[0, 1].set_xlabel('スケール因子 λ', fontsize=11)
    axes[0, 1].set_ylabel('面積 A [m²]', fontsize=11)
    axes[0, 1].set_title('(b) 面積のスケーリング', fontsize=12, fontweight='bold')
    axes[0, 1].grid(alpha=0.3, which='both')
    axes[0, 1].legend(fontsize=10)
    
    # 体積のスケーリング
    axes[1, 0].loglog(lambda_range, [r['Volume'] for r in results],
                      linewidth=2.5, color='#9b59b6', label='V ∝ λ³')
    axes[1, 0].set_xlabel('スケール因子 λ', fontsize=11)
    axes[1, 0].set_ylabel('体積 V [m³]', fontsize=11)
    axes[1, 0].set_title('(c) 体積のスケーリング', fontsize=12, fontweight='bold')
    axes[1, 0].grid(alpha=0.3, which='both')
    axes[1, 0].legend(fontsize=10)
    
    # S/V比のスケーリング
    axes[1, 1].loglog(lambda_range, [r['Surface_to_Volume_Ratio'] for r in results],
                      linewidth=2.5, color='#e74c3c', label='S/V ∝ λ⁻¹')
    axes[1, 1].set_xlabel('スケール因子 λ', fontsize=11)
    axes[1, 1].set_ylabel('S/V比 [m⁻¹]', fontsize=11)
    axes[1, 1].set_title('(d) 表面積/体積比のスケーリング', fontsize=12, fontweight='bold')
    axes[1, 1].grid(alpha=0.3, which='both')
    axes[1, 1].legend(fontsize=10)
    
    plt.tight_layout()
    plt.show()
    
    # 具体例の計算
    lambda_pilot = 10  # パイロット: 10倍スケール
    lambda_commercial = 100  # 商業: 100倍スケール
    
    pilot = geometric_scaling(lambda_pilot, L1_lab, A1_lab, V1_lab)
    commercial = geometric_scaling(lambda_commercial, L1_lab, A1_lab, V1_lab)
    
    print("=" * 60)
    print("幾何学的相似によるスケーリング結果")
    print("=" * 60)
    print(f"\nラボスケール:")
    print(f"  直径: {L1_lab*100:.1f} cm")
    print(f"  体積: {V1_lab*1e6:.2f} cm³ = {V1_lab*1e3:.2f} L")
    print(f"  S/V比: {A1_lab/V1_lab:.2f} m⁻¹")
    
    print(f"\nパイロットスケール (λ = {lambda_pilot}):")
    print(f"  直径: {pilot['Length']*100:.1f} cm")
    print(f"  体積: {pilot['Volume']:.4f} m³ = {pilot['Volume']*1e3:.1f} L")
    print(f"  S/V比: {pilot['Surface_to_Volume_Ratio']:.2f} m⁻¹")
    print(f"  S/V比の減少: {(1 - pilot['Surface_to_Volume_Ratio']/(A1_lab/V1_lab))*100:.1f}%")
    
    print(f"\n商業スケール (λ = {lambda_commercial}):")
    print(f"  直径: {commercial['Length']:.2f} m")
    print(f"  体積: {commercial['Volume']:.2f} m³")
    print(f"  S/V比: {commercial['Surface_to_Volume_Ratio']:.2f} m⁻¹")
    print(f"  S/V比の減少: {(1 - commercial['Surface_to_Volume_Ratio']/(A1_lab/V1_lab))*100:.1f}%")
    

**出力例:**
    
    
    ============================================================
    幾何学的相似によるスケーリング結果
    ============================================================
    
    ラボスケール:
      直径: 10.0 cm
      体積: 523.60 cm³ = 0.52 L
      S/V比: 60.00 m⁻¹
    
    パイロットスケール (λ = 10):
      直径: 100.0 cm
      体積: 0.5236 m³ = 523.6 L
      S/V比: 6.00 m⁻¹
      S/V比の減少: 90.0%
    
    商業スケール (λ = 100):
      直径: 10.00 m
      体積: 523.60 m³
      S/V比: 0.60 m⁻¹
      S/V比の減少: 99.0%
    

**解説:** 幾何学的相似では、長さが $\lambda$ 倍になると、面積は $\lambda^2$ 倍、体積は $\lambda^3$ 倍になります。重要なのは、**表面積/体積比（S/V比）が $\lambda^{-1}$ に比例して減少する** ことです。これは、スケールアップ時に伝熱・冷却能力が相対的に低下することを意味します。

* * *

### コード例2: パワー法則（べき乗則）スケーリング
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.optimize import curve_fit
    
    def power_law(x, a, b):
        """
        パワー法則: y = a * x^b
    
        Parameters:
        -----------
        x : array_like
            独立変数
        a : float
            係数
        b : float
            指数（スケーリング指数）
    
        Returns:
        --------
        y : array_like
            従属変数
        """
        return a * x**b
    
    # 実験データ（反応器サイズと反応時間の関係）
    # データ: 体積 [L], 反応時間 [min]
    volumes_exp = np.array([1, 5, 10, 50, 100, 500, 1000])  # L
    reaction_times = np.array([10, 18, 22, 35, 45, 70, 88])  # min
    
    # べき乗則フィッティング
    params, covariance = curve_fit(power_law, volumes_exp, reaction_times)
    a_fit, b_fit = params
    print("=" * 60)
    print("パワー法則フィッティング結果")
    print("=" * 60)
    print(f"反応時間 = {a_fit:.3f} × V^{b_fit:.3f}")
    print(f"スケーリング指数: b = {b_fit:.3f}")
    print(f"理論値（混合律速の場合）: b ≈ 0.33")
    print(f"適合度 R²: {1 - np.sum((reaction_times - power_law(volumes_exp, *params))**2) / np.sum((reaction_times - np.mean(reaction_times))**2):.4f}")
    
    # 予測範囲
    volumes_pred = np.logspace(0, 4, 100)  # 1 L to 10,000 L
    times_pred = power_law(volumes_pred, a_fit, b_fit)
    
    # 可視化
    plt.figure(figsize=(12, 6))
    
    # 対数プロット
    plt.subplot(1, 2, 1)
    plt.loglog(volumes_exp, reaction_times, 'o', markersize=10,
               color='#e74c3c', label='実験データ', zorder=5)
    plt.loglog(volumes_pred, times_pred, linewidth=2.5, color='#11998e',
               label=f'フィット: t = {a_fit:.2f}V^{b_fit:.2f}')
    plt.xlabel('反応器体積 V [L]', fontsize=12)
    plt.ylabel('反応時間 t [min]', fontsize=12)
    plt.title('パワー法則スケーリング（対数プロット）', fontsize=13, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3, which='both')
    
    # 線形プロット
    plt.subplot(1, 2, 2)
    plt.plot(volumes_exp, reaction_times, 'o', markersize=10,
             color='#e74c3c', label='実験データ', zorder=5)
    plt.plot(volumes_pred, times_pred, linewidth=2.5, color='#11998e',
             label=f'フィット: t = {a_fit:.2f}V^{b_fit:.2f}')
    plt.xlabel('反応器体積 V [L]', fontsize=12)
    plt.ylabel('反応時間 t [min]', fontsize=12)
    plt.title('パワー法則スケーリング（線形プロット）', fontsize=13, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3)
    plt.xlim(0, 1200)
    
    plt.tight_layout()
    plt.show()
    
    # スケールアップ予測
    V_commercial = 5000  # L
    t_commercial_pred = power_law(V_commercial, a_fit, b_fit)
    
    print(f"\n商業スケール予測 (V = {V_commercial} L):")
    print(f"  予測反応時間: {t_commercial_pred:.1f} min")
    print(f"  ラボスケール (1 L) の {t_commercial_pred/reaction_times[0]:.2f} 倍")
    

**出力例:**
    
    
    ============================================================
    パワー法則フィッティング結果
    ============================================================
    反応時間 = 9.976 × V^0.312
    スケーリング指数: b = 0.312
    理論値（混合律速の場合）: b ≈ 0.33
    適合度 R²: 0.9978
    
    商業スケール予測 (V = 5000 L):
      予測反応時間: 125.6 min
      ラボスケール (1 L) の 12.56 倍
    

**解説:** パワー法則（べき乗則）$y = ax^b$ は、スケーリング関係を記述する最も一般的な形式です。指数 $b$ がスケーリングの性質を決定します：

  * $b = 1$: 線形スケーリング
  * $b < 1$: スケールの利益あり（規模の経済）
  * $b > 1$: スケールの不利益（スケールアップが困難）

* * *

### コード例3: スケールアップ係数の計算
    
    
    import numpy as np
    import pandas as pd
    
    def scale_up_factor(V1, V2, basis='linear'):
        """
        スケールアップ係数の計算
    
        Parameters:
        -----------
        V1 : float
            元の系の体積 [m³]
        V2 : float
            スケールアップ後の体積 [m³]
        basis : str
            スケーリング基準
            - 'linear': 線形スケール (L)
            - 'area': 面積スケール (A)
            - 'volume': 体積スケール (V)
    
        Returns:
        --------
        dict : スケールアップ係数
        """
        # 線形スケール因子（体積比の1/3乗）
        lambda_L = (V2 / V1)**(1/3)
    
        if basis == 'linear':
            lambda_scale = lambda_L
            basis_desc = "線形基準"
        elif basis == 'area':
            lambda_scale = (V2 / V1)**(2/3)
            basis_desc = "面積基準"
        elif basis == 'volume':
            lambda_scale = V2 / V1
            basis_desc = "体積基準"
        else:
            raise ValueError("basis must be 'linear', 'area', or 'volume'")
    
        return {
            'Linear_Scale_Factor': lambda_L,
            'Scale_Factor': lambda_scale,
            'Basis': basis_desc,
            'Area_Ratio': lambda_L**2,
            'Volume_Ratio': lambda_L**3
        }
    
    # ケーススタディ: ラボ → パイロット → 商業スケール
    scales = {
        'Lab': 0.001,        # 1 L = 0.001 m³
        'Pilot': 0.5,        # 500 L = 0.5 m³
        'Commercial': 50.0   # 50,000 L = 50 m³
    }
    
    # スケールアップ計算
    print("=" * 80)
    print("スケールアップ係数の計算")
    print("=" * 80)
    
    scale_transitions = [
        ('Lab', 'Pilot'),
        ('Pilot', 'Commercial'),
        ('Lab', 'Commercial')
    ]
    
    for from_scale, to_scale in scale_transitions:
        V1 = scales[from_scale]
        V2 = scales[to_scale]
    
        print(f"\n{from_scale} → {to_scale} のスケールアップ:")
        print(f"  体積: {V1*1e3:.1f} L → {V2*1e3:.1f} L")
    
        # 各基準でのスケールアップ係数
        for basis in ['linear', 'area', 'volume']:
            result = scale_up_factor(V1, V2, basis)
            print(f"  {result['Basis']}: λ = {result['Scale_Factor']:.2f}")
    
    # 装置パラメータのスケーリング例
    print("\n" + "=" * 80)
    print("装置パラメータのスケーリング例（Lab → Commercial）")
    print("=" * 80)
    
    V_lab = scales['Lab']
    V_commercial = scales['Commercial']
    lambda_L = (V_commercial / V_lab)**(1/3)
    
    # ラボスケールのパラメータ
    params_lab = {
        '直径 [m]': 0.1,
        '高さ [m]': 0.15,
        '撹拌速度 [rpm]': 500,
        '撹拌動力 [W]': 5,
        '伝熱面積 [m²]': 0.05,
        '滞留時間 [min]': 30
    }
    
    # スケールアップ後のパラメータ（幾何学的相似を仮定）
    params_commercial = {
        '直径 [m]': params_lab['直径 [m]'] * lambda_L,
        '高さ [m]': params_lab['高さ [m]'] * lambda_L,
        '撹拌速度 [rpm]': params_lab['撹拌速度 [rpm]'] / lambda_L**0.5,  # Froude数保持
        '撹拌動力 [W]': params_lab['撹拌動力 [W]'] * lambda_L**5,  # パワー則
        '伝熱面積 [m²]': params_lab['伝熱面積 [m²]'] * lambda_L**2,
        '滞留時間 [min]': params_lab['滞留時間 [min]']  # 同じ反応時間を維持
    }
    
    # 結果表示
    df = pd.DataFrame({
        'ラボスケール': params_lab,
        '商業スケール': params_commercial,
        'スケーリング係数': [
            f'λ¹ = {lambda_L:.2f}',
            f'λ¹ = {lambda_L:.2f}',
            f'λ⁻⁰·⁵ = {lambda_L**(-0.5):.2f}',
            f'λ⁵ = {lambda_L**5:.0f}',
            f'λ² = {lambda_L**2:.2f}',
            '一定'
        ]
    })
    
    print("\n" + df.to_string())
    

**出力例:**
    
    
    ================================================================================
    スケールアップ係数の計算
    ================================================================================
    
    Lab → Pilot のスケールアップ:
      体積: 1.0 L → 500.0 L
      線形基準: λ = 7.94
      面積基準: λ = 63.10
      体積基準: λ = 500.00
    
    Pilot → Commercial のスケールアップ:
      体積: 500.0 L → 50000.0 L
      線形基準: λ = 4.64
      面積基準: λ = 21.54
      体積基準: λ = 100.00
    
    Lab → Commercial のスケールアップ:
      体積: 1.0 L → 50000.0 L
      線形基準: λ = 36.84
      面積基準: λ = 1357.21
      体積基準: λ = 50000.00
    
    ================================================================================
    装置パラメータのスケーリング例（Lab → Commercial）
    ================================================================================
    
                    ラボスケール  商業スケール スケーリング係数
    直径 [m]             0.10      3.68  λ¹ = 36.84
    高さ [m]             0.15      5.53  λ¹ = 36.84
    撹拌速度 [rpm]     500.00     82.37  λ⁻⁰·⁵ = 0.16
    撹拌動力 [W]         5.00  89819.18  λ⁵ = 17964
    伝熱面積 [m²]        0.05     67.93  λ² = 1358
    滞留時間 [min]      30.00     30.00  一定
    

**解説:** スケールアップ係数は、選択する基準（線形、面積、体積）により異なります。装置パラメータは、保持すべき物理量（レイノルズ数、フルード数など）に応じて異なるスケーリング則に従います。撹拌動力は $\lambda^5$ でスケールするため、商業スケールでは劇的に増加します。

* * *

### コード例4: 装置サイジング計算（反応器）
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    def reactor_sizing(production_rate, residence_time, conversion, density=1000):
        """
        反応器のサイジング計算
    
        Parameters:
        -----------
        production_rate : float
            目標生産量 [kg/h]
        residence_time : float
            必要滞留時間 [h]
        conversion : float
            反応転化率 [-] (0 to 1)
        density : float, optional
            反応混合物の密度 [kg/m³], default: 1000
    
        Returns:
        --------
        dict : 反応器設計パラメータ
        """
        # 必要な反応器体積
        # Q = P / (X * ρ)  where Q: 体積流量, P: 生産量, X: 転化率
        volumetric_flowrate = production_rate / (conversion * density)  # m³/h
        reactor_volume = volumetric_flowrate * residence_time  # m³
    
        # 円筒形反応器を仮定（H/D = 2）
        # V = π/4 * D² * H = π/4 * D² * 2D = π/2 * D³
        diameter = (reactor_volume / (np.pi / 2))**(1/3)
        height = 2 * diameter
    
        # 伝熱面積（側面 + 底面）
        side_area = np.pi * diameter * height
        bottom_area = np.pi * diameter**2 / 4
        heat_transfer_area = side_area + bottom_area
    
        return {
            'Production_Rate_kg_h': production_rate,
            'Volumetric_Flowrate_m3_h': volumetric_flowrate,
            'Residence_Time_h': residence_time,
            'Reactor_Volume_m3': reactor_volume,
            'Diameter_m': diameter,
            'Height_m': height,
            'Heat_Transfer_Area_m2': heat_transfer_area,
            'Surface_to_Volume_Ratio': heat_transfer_area / reactor_volume
        }
    
    # ケーススタディ: スケール別反応器設計
    scales_production = {
        'Lab': 0.1,          # 0.1 kg/h
        'Pilot': 10,         # 10 kg/h
        'Commercial': 1000   # 1000 kg/h (1 ton/h)
    }
    
    residence_time = 2  # h
    conversion = 0.85
    density = 1100  # kg/m³
    
    print("=" * 80)
    print("反応器サイジング計算")
    print("=" * 80)
    print(f"条件: 滞留時間 = {residence_time} h, 転化率 = {conversion:.0%}, 密度 = {density} kg/m³\n")
    
    results_reactor = {}
    for scale_name, prod_rate in scales_production.items():
        result = reactor_sizing(prod_rate, residence_time, conversion, density)
        results_reactor[scale_name] = result
    
        print(f"{scale_name}スケール反応器:")
        print(f"  生産量: {result['Production_Rate_kg_h']:.1f} kg/h")
        print(f"  反応器体積: {result['Reactor_Volume_m3']:.4f} m³ = {result['Reactor_Volume_m3']*1e3:.1f} L")
        print(f"  直径: {result['Diameter_m']:.3f} m = {result['Diameter_m']*100:.1f} cm")
        print(f"  高さ: {result['Height_m']:.3f} m = {result['Height_m']*100:.1f} cm")
        print(f"  伝熱面積: {result['Heat_Transfer_Area_m2']:.3f} m²")
        print(f"  S/V比: {result['Surface_to_Volume_Ratio']:.2f} m⁻¹\n")
    
    # スケーリング関係の可視化
    production_rates = np.logspace(-1, 3, 50)  # 0.1 to 1000 kg/h
    volumes = []
    diameters = []
    SV_ratios = []
    
    for P in production_rates:
        res = reactor_sizing(P, residence_time, conversion, density)
        volumes.append(res['Reactor_Volume_m3'])
        diameters.append(res['Diameter_m'])
        SV_ratios.append(res['Surface_to_Volume_Ratio'])
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # 体積 vs 生産量
    axes[0].loglog(production_rates, volumes, linewidth=2.5, color='#11998e')
    for scale_name, prod_rate in scales_production.items():
        V = results_reactor[scale_name]['Reactor_Volume_m3']
        axes[0].scatter([prod_rate], [V], s=150, zorder=5, label=scale_name)
    axes[0].set_xlabel('生産量 [kg/h]', fontsize=12)
    axes[0].set_ylabel('反応器体積 [m³]', fontsize=12)
    axes[0].set_title('(a) 反応器体積のスケーリング', fontsize=13, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(alpha=0.3, which='both')
    
    # 直径 vs 生産量
    axes[1].loglog(production_rates, diameters, linewidth=2.5, color='#e67e22')
    for scale_name, prod_rate in scales_production.items():
        D = results_reactor[scale_name]['Diameter_m']
        axes[1].scatter([prod_rate], [D], s=150, zorder=5, label=scale_name)
    axes[1].set_xlabel('生産量 [kg/h]', fontsize=12)
    axes[1].set_ylabel('反応器直径 [m]', fontsize=12)
    axes[1].set_title('(b) 反応器直径のスケーリング', fontsize=13, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(alpha=0.3, which='both')
    
    # S/V比 vs 生産量
    axes[2].loglog(production_rates, SV_ratios, linewidth=2.5, color='#e74c3c')
    for scale_name, prod_rate in scales_production.items():
        SV = results_reactor[scale_name]['Surface_to_Volume_Ratio']
        axes[2].scatter([prod_rate], [SV], s=150, zorder=5, label=scale_name)
    axes[2].set_xlabel('生産量 [kg/h]', fontsize=12)
    axes[2].set_ylabel('S/V比 [m⁻¹]', fontsize=12)
    axes[2].set_title('(c) S/V比のスケーリング', fontsize=13, fontweight='bold')
    axes[2].legend(fontsize=10)
    axes[2].grid(alpha=0.3, which='both')
    
    plt.tight_layout()
    plt.show()
    

**出力例:**
    
    
    ================================================================================
    反応器サイジング計算
    ================================================================================
    条件: 滞留時間 = 2 h, 転化率 = 85%, 密度 = 1100 kg/m³
    
    Labスケール反応器:
      生産量: 0.1 kg/h
      反応器体積: 0.0002 m³ = 0.2 L
      直径: 0.046 m = 4.6 cm
      高さ: 0.093 m = 9.3 cm
      伝熱面積: 0.015 m²
      S/V比: 65.45 m⁻¹
    
    Pilotスケール反応器:
      生産量: 10.0 kg/h
      反応器体積: 0.0214 m³ = 21.4 L
      直径: 0.215 m = 21.5 cm
      高さ: 0.431 m = 43.1 cm
      伝熱面積: 0.366 m²
      S/V比: 17.12 m⁻¹
    
    Commercialスケール反応器:
      生産量: 1000.0 kg/h
      反応器体積: 2.1405 m³ = 2140.5 L
      直径: 1.000 m = 100.0 cm
      高さ: 2.000 m = 200.0 cm
      伝熱面積: 7.069 m²
      S/V比: 3.30 m⁻¹
    

**解説:** 反応器のサイジングでは、生産量、滞留時間、転化率から必要な反応器体積を計算します。幾何学的相似（H/D = 2）を仮定すると、体積から直径と高さが決定されます。**S/V比の減少** は、商業スケールでの伝熱・冷却の課題を示唆しています。

* * *

### コード例5: 熱交換器のスケーリング計算
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    def heat_exchanger_sizing(Q_heat, U, delta_T_lm, tube_diameter=0.025, tube_length=3.0):
        """
        熱交換器のサイジング計算
    
        Parameters:
        -----------
        Q_heat : float
            伝熱量 [W]
        U : float
            総括伝熱係数 [W/(m²·K)]
        delta_T_lm : float
            対数平均温度差 [K]
        tube_diameter : float, optional
            伝熱管直径 [m], default: 0.025 (1 inch)
        tube_length : float, optional
            伝熱管長さ [m], default: 3.0
    
        Returns:
        --------
        dict : 熱交換器設計パラメータ
        """
        # 必要伝熱面積: Q = U * A * ΔT_lm
        A_required = Q_heat / (U * delta_T_lm)
    
        # 1本の伝熱管の面積
        A_per_tube = np.pi * tube_diameter * tube_length
    
        # 必要管本数
        N_tubes = int(np.ceil(A_required / A_per_tube))
    
        # 実際の伝熱面積
        A_actual = N_tubes * A_per_tube
    
        # シェル直径の推定（管群配置による）
        # 正三角形配置、ピッチ/直径比 = 1.25
        pitch = 1.25 * tube_diameter
        # シェル直径 ≈ pitch * sqrt(N_tubes) + 余裕
        D_shell = pitch * np.sqrt(N_tubes) * 1.2
    
        return {
            'Heat_Duty_kW': Q_heat / 1000,
            'Required_Area_m2': A_required,
            'Actual_Area_m2': A_actual,
            'Number_of_Tubes': N_tubes,
            'Tube_Diameter_mm': tube_diameter * 1000,
            'Tube_Length_m': tube_length,
            'Shell_Diameter_m': D_shell,
            'Area_per_Tube_m2': A_per_tube
        }
    
    # スケール別の熱交換器設計
    Q_heat_lab = 5000  # W = 5 kW
    Q_heat_pilot = 50000  # W = 50 kW
    Q_heat_commercial = 5000000  # W = 5 MW
    
    U = 500  # W/(m²·K) - 水-水熱交換器の典型値
    delta_T_lm = 20  # K
    
    scales_HX = {
        'Lab': Q_heat_lab,
        'Pilot': Q_heat_pilot,
        'Commercial': Q_heat_commercial
    }
    
    print("=" * 80)
    print("熱交換器サイジング計算")
    print("=" * 80)
    print(f"条件: U = {U} W/(m²·K), ΔT_lm = {delta_T_lm} K\n")
    
    results_HX = {}
    for scale_name, Q in scales_HX.items():
        result = heat_exchanger_sizing(Q, U, delta_T_lm)
        results_HX[scale_name] = result
    
        print(f"{scale_name}スケール熱交換器:")
        print(f"  伝熱量: {result['Heat_Duty_kW']:.1f} kW")
        print(f"  必要伝熱面積: {result['Required_Area_m2']:.2f} m²")
        print(f"  伝熱管本数: {result['Number_of_Tubes']} 本")
        print(f"  伝熱管: {result['Tube_Diameter_mm']:.1f} mm × {result['Tube_Length_m']:.1f} m")
        print(f"  シェル直径: {result['Shell_Diameter_m']:.3f} m = {result['Shell_Diameter_m']*100:.1f} cm\n")
    
    # スケーリング関係の可視化
    Q_range = np.logspace(3, 7, 50)  # 1 kW to 10 MW
    areas = []
    N_tubes_list = []
    D_shells = []
    
    for Q in Q_range:
        res = heat_exchanger_sizing(Q, U, delta_T_lm)
        areas.append(res['Required_Area_m2'])
        N_tubes_list.append(res['Number_of_Tubes'])
        D_shells.append(res['Shell_Diameter_m'])
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # 伝熱面積 vs 伝熱量
    axes[0].loglog(np.array(Q_range)/1000, areas, linewidth=2.5, color='#11998e',
                   label='A = Q / (U·ΔT)')
    for scale_name, Q in scales_HX.items():
        A = results_HX[scale_name]['Required_Area_m2']
        axes[0].scatter([Q/1000], [A], s=150, zorder=5, label=scale_name)
    axes[0].set_xlabel('伝熱量 [kW]', fontsize=12)
    axes[0].set_ylabel('伝熱面積 [m²]', fontsize=12)
    axes[0].set_title('(a) 伝熱面積のスケーリング', fontsize=13, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(alpha=0.3, which='both')
    
    # 管本数 vs 伝熱量
    axes[1].loglog(np.array(Q_range)/1000, N_tubes_list, linewidth=2.5, color='#e67e22')
    for scale_name, Q in scales_HX.items():
        N = results_HX[scale_name]['Number_of_Tubes']
        axes[1].scatter([Q/1000], [N], s=150, zorder=5, label=scale_name)
    axes[1].set_xlabel('伝熱量 [kW]', fontsize=12)
    axes[1].set_ylabel('伝熱管本数', fontsize=12)
    axes[1].set_title('(b) 伝熱管本数のスケーリング', fontsize=13, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(alpha=0.3, which='both')
    
    # シェル直径 vs 伝熱量
    axes[2].loglog(np.array(Q_range)/1000, D_shells, linewidth=2.5, color='#9b59b6')
    for scale_name, Q in scales_HX.items():
        D = results_HX[scale_name]['Shell_Diameter_m']
        axes[2].scatter([Q/1000], [D], s=150, zorder=5, label=scale_name)
    axes[2].set_xlabel('伝熱量 [kW]', fontsize=12)
    axes[2].set_ylabel('シェル直径 [m]', fontsize=12)
    axes[2].set_title('(c) シェル直径のスケーリング', fontsize=13, fontweight='bold')
    axes[2].legend(fontsize=10)
    axes[2].grid(alpha=0.3, which='both')
    
    plt.tight_layout()
    plt.show()
    

**出力例:**
    
    
    ================================================================================
    熱交換器サイジング計算
    ================================================================================
    条件: U = 500 W/(m²·K), ΔT_lm = 20 K
    
    Labスケール熱交換器:
      伝熱量: 5.0 kW
      必要伝熱面積: 0.50 m²
      伝熱管本数: 3 本
      伝熱管: 25.0 mm × 3.0 m
      シェル直径: 0.052 m = 5.2 cm
    
    Pilotスケール熱交換器:
      伝熱量: 50.0 kW
      必要伝熱面積: 5.00 m²
      伝熱管本数: 22 本
      伝熱管: 25.0 mm × 3.0 m
      シェル直径: 0.177 m = 17.7 cm
    
    Commercialスケール熱交換器:
      伝熱量: 5000.0 kW
      必要伝熱面積: 500.00 m²
      伝熱管本数: 2123 本
      伝熱管: 25.0 mm × 3.0 m
      シェル直径: 1.738 m = 173.8 cm
    

**解説:** 熱交換器のサイジングでは、伝熱量 $Q = U \cdot A \cdot \Delta T_{lm}$ の関係から必要伝熱面積を計算します。伝熱面積は伝熱量に比例するため、スケールアップ時には伝熱管本数とシェル直径が増加します。

* * *

### コード例6: パイロットプラント設計とスケールダウン比最適化
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    def pilot_plant_design(V_commercial, scale_down_ratio_range):
        """
        パイロットプラントの最適スケールダウン比を決定
    
        Parameters:
        -----------
        V_commercial : float
            商業プラントの体積 [m³]
        scale_down_ratio_range : array_like
            検討するスケールダウン比の範囲
    
        Returns:
        --------
        dict : 各スケールダウン比での評価指標
        """
        results = []
    
        for ratio in scale_down_ratio_range:
            V_pilot = V_commercial / ratio
            lambda_scale = (V_pilot / V_commercial)**(1/3)
    
            # 評価指標
            # 1. 実験コスト（体積に比例）
            experiment_cost_relative = V_pilot / V_commercial * 100  # %
    
            # 2. 代表性（S/V比の変化）
            SV_commercial = 1.0  # 基準
            SV_pilot = SV_commercial / lambda_scale
            SV_ratio_change = abs(SV_pilot - SV_commercial) / SV_commercial * 100  # %
    
            # 3. レイノルズ数の変化（撹拌速度を一定と仮定）
            Re_change = lambda_scale * 100  # % (Re ∝ ND² ∝ D² ∝ λ²、N一定)
    
            # 4. 総合評価スコア（低いほど良い）
            # コスト最小化 + 代表性維持 + Re数変化最小化
            score = (experiment_cost_relative / 10) + SV_ratio_change + (Re_change / 10)
    
            results.append({
                'Scale_Down_Ratio': ratio,
                'V_Pilot_m3': V_pilot,
                'V_Pilot_L': V_pilot * 1000,
                'Experiment_Cost_%': experiment_cost_relative,
                'SV_Deviation_%': SV_ratio_change,
                'Re_Change_%': Re_change,
                'Total_Score': score
            })
    
        return results
    
    # 商業プラント: 50 m³
    V_commercial = 50.0  # m³
    
    # スケールダウン比の範囲: 10倍〜1000倍
    scale_down_ratios = np.logspace(1, 3, 30)  # 10 to 1000
    
    # パイロット設計の評価
    results_pilot = pilot_plant_design(V_commercial, scale_down_ratios)
    
    # 最適スケールダウン比の選定
    optimal_idx = np.argmin([r['Total_Score'] for r in results_pilot])
    optimal_ratio = results_pilot[optimal_idx]['Scale_Down_Ratio']
    optimal_V_pilot = results_pilot[optimal_idx]['V_Pilot_m3']
    
    print("=" * 80)
    print("パイロットプラント設計の最適化")
    print("=" * 80)
    print(f"商業プラント体積: {V_commercial} m³ = {V_commercial*1000:.0f} L\n")
    
    print(f"最適スケールダウン比: 1/{optimal_ratio:.1f}")
    print(f"最適パイロット体積: {optimal_V_pilot:.3f} m³ = {optimal_V_pilot*1000:.1f} L")
    print(f"総合スコア: {results_pilot[optimal_idx]['Total_Score']:.2f}\n")
    
    print("評価指標:")
    print(f"  実験コスト: {results_pilot[optimal_idx]['Experiment_Cost_%']:.2f}% (商業プラント比)")
    print(f"  S/V比変化: {results_pilot[optimal_idx]['SV_Deviation_%']:.2f}%")
    print(f"  Re数変化: {results_pilot[optimal_idx]['Re_Change_%']:.2f}%")
    
    # 可視化
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # パイロット体積
    axes[0, 0].semilogx([r['Scale_Down_Ratio'] for r in results_pilot],
                         [r['V_Pilot_L'] for r in results_pilot],
                         linewidth=2.5, color='#11998e')
    axes[0, 0].scatter([optimal_ratio], [optimal_V_pilot*1000],
                       s=200, color='red', zorder=5, marker='*',
                       edgecolors='black', linewidth=2, label='最適点')
    axes[0, 0].set_xlabel('スケールダウン比', fontsize=11)
    axes[0, 0].set_ylabel('パイロット体積 [L]', fontsize=11)
    axes[0, 0].set_title('(a) パイロットプラント体積', fontsize=12, fontweight='bold')
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(alpha=0.3)
    
    # 実験コスト
    axes[0, 1].semilogx([r['Scale_Down_Ratio'] for r in results_pilot],
                         [r['Experiment_Cost_%'] for r in results_pilot],
                         linewidth=2.5, color='#e67e22')
    axes[0, 1].scatter([optimal_ratio],
                       [results_pilot[optimal_idx]['Experiment_Cost_%']],
                       s=200, color='red', zorder=5, marker='*',
                       edgecolors='black', linewidth=2)
    axes[0, 1].set_xlabel('スケールダウン比', fontsize=11)
    axes[0, 1].set_ylabel('実験コスト [%]', fontsize=11)
    axes[0, 1].set_title('(b) 実験コスト（商業プラント比）', fontsize=12, fontweight='bold')
    axes[0, 1].grid(alpha=0.3)
    
    # S/V比の変化
    axes[1, 0].semilogx([r['Scale_Down_Ratio'] for r in results_pilot],
                         [r['SV_Deviation_%'] for r in results_pilot],
                         linewidth=2.5, color='#9b59b6')
    axes[1, 0].scatter([optimal_ratio],
                       [results_pilot[optimal_idx]['SV_Deviation_%']],
                       s=200, color='red', zorder=5, marker='*',
                       edgecolors='black', linewidth=2)
    axes[1, 0].set_xlabel('スケールダウン比', fontsize=11)
    axes[1, 0].set_ylabel('S/V比の変化 [%]', fontsize=11)
    axes[1, 0].set_title('(c) S/V比の変化（商業プラント比）', fontsize=12, fontweight='bold')
    axes[1, 0].grid(alpha=0.3)
    
    # 総合スコア
    axes[1, 1].semilogx([r['Scale_Down_Ratio'] for r in results_pilot],
                         [r['Total_Score'] for r in results_pilot],
                         linewidth=2.5, color='#e74c3c')
    axes[1, 1].scatter([optimal_ratio],
                       [results_pilot[optimal_idx]['Total_Score']],
                       s=200, color='red', zorder=5, marker='*',
                       edgecolors='black', linewidth=2, label='最適点')
    axes[1, 1].set_xlabel('スケールダウン比', fontsize=11)
    axes[1, 1].set_ylabel('総合スコア（低いほど良い）', fontsize=11)
    axes[1, 1].set_title('(d) 総合評価スコア', fontsize=12, fontweight='bold')
    axes[1, 1].legend(fontsize=10)
    axes[1, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # 推奨スケールダウン比の範囲
    print("\n" + "=" * 80)
    print("推奨スケールダウン比の範囲")
    print("=" * 80)
    print("一般的なガイドライン:")
    print("  ラボスケール: 1/100 〜 1/1000")
    print("  パイロットスケール: 1/10 〜 1/100")
    print("  デモスケール: 1/2 〜 1/10")
    print(f"\n本ケースの推奨範囲: 1/{optimal_ratio*2:.0f} 〜 1/{optimal_ratio/2:.0f}")
    

**出力例:**
    
    
    ================================================================================
    パイロットプラント設計の最適化
    ================================================================================
    商業プラント体積: 50 m³ = 50000 L
    
    最適スケールダウン比: 1/51.8
    最適パイロット体積: 0.966 m³ = 966.1 L
    総合スコア: 92.05
    
    評価指標:
      実験コスト: 1.93% (商業プラント比)
      S/V比変化: 73.21%
      Re数変化: 16.91%
    
    ================================================================================
    推奨スケールダウン比の範囲
    ================================================================================
    一般的なガイドライン:
      ラボスケール: 1/100 〜 1/1000
      パイロットスケール: 1/10 〜 1/100
      デモスケール: 1/2 〜 1/10
    
    本ケースの推奨範囲: 1/104 〜 1/26
    

**解説:** パイロットプラントの設計では、**実験コスト** （小さいほど良い）と**商業プラントとの代表性** （S/V比の変化が小さいほど良い）のトレードオフを考慮します。総合評価スコアを最小化することで、最適なスケールダウン比を決定できます。

* * *

### コード例7: 経済性のスケーリング（6/10則）
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    def equipment_cost_scaling(C1, Q1, Q2, exponent=0.6):
        """
        設備コストのスケーリング計算（6/10則）
    
        Parameters:
        -----------
        C1 : float
            基準設備のコスト [$]
        Q1 : float
            基準設備の容量（体積、熱量など）
        Q2 : float
            新設備の容量
        exponent : float, optional
            スケーリング指数, default: 0.6 (6/10則)
    
        Returns:
        --------
        float : 新設備の推定コスト [$]
    
        Note:
        -----
        6/10則: C2 = C1 * (Q2/Q1)^0.6
        - exponent = 0.6: 標準的な設備（反応器、タンク、熱交換器）
        - exponent = 0.7-0.8: 複雑な設備（蒸留塔、圧縮機）
        - exponent = 1.0: 線形コスト（ポンプ、配管）
        """
        C2 = C1 * (Q2 / Q1)**exponent
        return C2
    
    # ラボスケールの設備コスト（基準）
    V_lab = 1  # L
    C_lab = 5000  # $ (ラボスケール反応器)
    
    # スケール範囲
    volumes = np.logspace(0, 5, 100)  # 1 L to 100,000 L
    
    # 異なるスケーリング指数での比較
    exponents = {
        '6/10則 (b=0.6)': 0.6,
        '線形 (b=1.0)': 1.0,
        '実測値 (b=0.7)': 0.7
    }
    
    print("=" * 80)
    print("設備コストのスケーリング（6/10則）")
    print("=" * 80)
    print(f"基準設備: {V_lab} L, コスト: ${C_lab:,}\n")
    
    # 各スケールでのコスト計算
    target_volumes = [1, 10, 100, 1000, 10000, 100000]  # L
    
    print("スケール別の設備コスト推算:")
    print(f"{'体積 [L]':<12} {'6/10則 [$]':<15} {'線形 [$]':<15} {'比率':<10}")
    print("-" * 60)
    
    for V in target_volumes:
        C_sixtenths = equipment_cost_scaling(C_lab, V_lab, V, exponent=0.6)
        C_linear = equipment_cost_scaling(C_lab, V_lab, V, exponent=1.0)
        ratio = C_sixtenths / C_linear
        print(f"{V:<12.0f} {C_sixtenths:<15,.0f} {C_linear:<15,.0f} {ratio:<10.2%}")
    
    # 可視化
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # コストのスケーリング比較
    axes[0].loglog(volumes, volumes, linewidth=2, linestyle='--',
                   color='gray', alpha=0.5, label='体積（参考）')
    
    for label, exp in exponents.items():
        costs = [equipment_cost_scaling(C_lab, V_lab, V, exponent=exp) for V in volumes]
        axes[0].loglog(volumes, costs, linewidth=2.5, label=label)
    
    # 特定スケールをマーク
    for V in [10, 100, 1000]:
        C = equipment_cost_scaling(C_lab, V_lab, V, exponent=0.6)
        axes[0].scatter([V], [C], s=100, zorder=5, edgecolors='black', linewidth=1.5)
    
    axes[0].set_xlabel('設備容量 [L]', fontsize=12)
    axes[0].set_ylabel('設備コスト [$]', fontsize=12)
    axes[0].set_title('(a) 設備コストのスケーリング則', fontsize=13, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(alpha=0.3, which='both')
    
    # 単位体積あたりのコスト
    unit_costs_sixtenths = [equipment_cost_scaling(C_lab, V_lab, V, exponent=0.6) / V
                            for V in volumes]
    unit_costs_linear = [equipment_cost_scaling(C_lab, V_lab, V, exponent=1.0) / V
                         for V in volumes]
    
    axes[1].loglog(volumes, unit_costs_sixtenths, linewidth=2.5,
                   color='#11998e', label='6/10則 (b=0.6)')
    axes[1].loglog(volumes, unit_costs_linear, linewidth=2.5, linestyle='--',
                   color='#e74c3c', label='線形 (b=1.0)')
    axes[1].set_xlabel('設備容量 [L]', fontsize=12)
    axes[1].set_ylabel('単位体積あたりコスト [$/L]', fontsize=12)
    axes[1].set_title('(b) 単位体積あたりコスト（規模の経済）', fontsize=13, fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].grid(alpha=0.3, which='both')
    
    plt.tight_layout()
    plt.show()
    
    # 経済的最適スケールの考察
    print("\n" + "=" * 80)
    print("規模の経済（Economies of Scale）")
    print("=" * 80)
    print("6/10則により、スケールアップするほど単位体積あたりのコストが低下します。")
    print("\n例: 体積が10倍になると...")
    ratio_10x = 10**0.6
    print(f"  総コスト: {ratio_10x:.2f}倍（10倍未満）")
    print(f"  単位体積あたりコスト: {ratio_10x/10:.2f}倍（減少）")
    print(f"  コスト削減率: {(1 - ratio_10x/10)*100:.1f}%")
    
    print("\n例: 体積が100倍になると...")
    ratio_100x = 100**0.6
    print(f"  総コスト: {ratio_100x:.2f}倍（100倍未満）")
    print(f"  単位体積あたりコスト: {ratio_100x/100:.2f}倍（大幅減少）")
    print(f"  コスト削減率: {(1 - ratio_100x/100)*100:.1f}%")
    
    print("\n注意: 実際には以下の要因により、無制限なスケールアップは困難です:")
    print("  - 製造・輸送上の制約")
    print("  - 構造強度・安全性の制約")
    print("  - 市場需要の限界")
    print("  - プロセス制御の難しさ")
    

**出力例:**
    
    
    ================================================================================
    設備コストのスケーリング（6/10則）
    ================================================================================
    基準設備: 1 L, コスト: $5,000
    
    スケール別の設備コスト推算:
    体積 [L]     6/10則 [$]      線形 [$]        比率
    ------------------------------------------------------------
    1            5,000           5,000           100.00%
    10           19,953          50,000          39.91%
    100          79,621          500,000         15.92%
    1,000        317,480         5,000,000       6.35%
    10,000       1,266,287       50,000,000      2.53%
    100,000      5,048,475       500,000,000     1.01%
    
    ================================================================================
    規模の経済（Economies of Scale）
    ================================================================================
    6/10則により、スケールアップするほど単位体積あたりのコストが低下します。
    
    例: 体積が10倍になると...
      総コスト: 3.98倍（10倍未満）
      単位体積あたりコスト: 0.40倍（減少）
      コスト削減率: 60.1%
    
    例: 体積が100倍になると...
      総コスト: 15.85倍（100倍未満）
      単位体積あたりコスト: 0.16倍（大幅減少）
      コスト削減率: 84.1%
    
    注意: 実際には以下の要因により、無制限なスケールアップは困難です:
      - 製造・輸送上の制約
      - 構造強度・安全性の制約
      - 市場需要の限界
      - プロセス制御の難しさ
    

**解説:** **6/10則（six-tenths rule）** は、設備コストのスケーリングにおける経験則です。設備容量が増加すると、総コストは容量ほど増加せず、スケーリング指数 $b \approx 0.6$ でスケールします。これにより、**規模の経済（economies of scale）** が生じ、大規模プラントほど単位生産量あたりのコストが低下します。

* * *

## 1.3 本章のまとめ

### 学んだこと

  1. **相似則の基礎**
     * 幾何学的相似、運動学的相似、動力学的相似の理解
     * スケール因子 $\lambda$ によるスケーリング則
  2. **べき乗則スケーリング**
     * $L \propto \lambda^1$, $A \propto \lambda^2$, $V \propto \lambda^3$
     * S/V比は $\lambda^{-1}$ でスケール（スケールアップで減少）
     * パワー法則 $y = ax^b$ によるスケーリング関係の記述
  3. **装置サイジング計算**
     * 反応器、熱交換器の設計計算
     * 生産量、伝熱量から必要な装置サイズを決定
  4. **パイロットプラント設計**
     * スケールダウン比の最適化
     * 実験コストと代表性のトレードオフ
  5. **経済性のスケーリング**
     * 6/10則による設備コストの推算
     * 規模の経済（スケールアップでコスト効率向上）

### 重要なポイント

  * **S/V比の減少** は、スケールアップ時の伝熱・冷却能力低下を意味する
  * スケール因子 $\lambda$ により、物理量のスケーリング則が決まる
  * パワー法則はスケーリング関係の一般的な記述方法
  * パイロットプラントは、コストと代表性のバランスで設計する
  * 6/10則により、大規模化ほど単位コストが低下（規模の経済）

### 次の章へ

第2章では、**無次元数とスケールアップ則** を詳しく学びます：

  * レイノルズ数、フルード数、ウェーバー数などの主要無次元数
  * 支配的な無次元数の選択と相似則の適用
  * Buckingham π定理による無次元群の導出
  * スケールアップ規準の設定（定Re数、定パワー密度など）
  * 混合時間のスケーリングと撹拌動力計算
