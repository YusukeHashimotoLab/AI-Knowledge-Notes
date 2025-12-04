---
title: 第3章 粒子・積層複合材料
chapter_title: 第3章 粒子・積層複合材料
---

🌐 JP | [🇬🇧 EN](<../../../en/MS/composite-materials-introduction/chapter-3.html>) | Last sync: 2025-11-16

[AI寺子屋トップ](<../../index.html>)›[材料科学](<../../MS/index.html>)›[Composite Materials](<../../MS/composite-materials-introduction/index.html>)›Chapter 3

### 複合材料入門

  * [目次](<index.html>)
  * [第1章 複合材料の基礎](<chapter-1.html>)
  * [第2章 繊維強化複合材料](<chapter-2.html>)
  * [第3章 粒子・積層複合材料](<chapter-3.html>)
  * 第4章 複合材料の評価（準備中）
  * [第5章 Python実践](<chapter-5.html>)

#### Materials Science シリーズ

  * [高分子材料入門](<../polymer-materials-introduction/index.html>)
  * 薄膜・ナノ材料入門 (準備中)
  * [複合材料入門](<index.html>)

# 第3章 粒子・積層複合材料

### 学習目標

  * **基礎レベル:** 粒子強化複合材料の種類と強化メカニズムを理解し、基本的な特性予測ができる
  * **応用レベル:** Orowanメカニズムを適用し、粒子サイズ・分率の最適化ができる
  * **発展レベル:** MMC/CMCの設計パラメータを総合的に評価し、用途に応じた材料選定ができる

## 3.1 粒子強化複合材料の基礎

### 3.1.1 粒子強化の分類

粒子強化複合材料は、母材中に粒子状の強化材を分散させた材料です。 強化メカニズムにより以下に分類されます： 

分類 | 粒子サイズ | 強化機構 | 代表例  
---|---|---|---  
分散強化 | 10-100 nm | 転位のバイパス(Orowan) | ODS合金、析出強化鋼  
粒子強化 | 1-100 μm | 荷重分担、熱膨張差 | SiC/Al, WC/Co  
充填材 | 1-100 μm | コスト低減、寸法安定性 | 炭酸カルシウム/樹脂  
      
    
    ```mermaid
    flowchart TD
                                A[粒子強化複合材料] --> B[金属基 MMC]
                                A --> C[セラミック基 CMC]
                                A --> D[樹脂基 PMC]
    
                                B --> E[SiC/Al自動車ピストン]
                                B --> F[Al2O3/Al摺動部品]
                                B --> G[B4C/Al装甲材]
    
                                C --> H[SiC/SiC耐熱部材]
                                C --> I[Al2O3/ZrO2切削工具]
    
                                D --> J[カーボンブラック/ゴムタイヤ]
                                D --> K[ガラスビーズ/樹脂電子基板]
    
                                style A fill:#e1f5ff
                                style E fill:#ffe1e1
                                style F fill:#ffe1e1
                                style G fill:#ffe1e1
                                style H fill:#c8e6c9
                                style I fill:#c8e6c9
                                style J fill:#fff9c4
                                style K fill:#fff9c4
    ```

### 3.1.2 MMC (Metal Matrix Composites)

金属を母材とする複合材料で、軽量・高強度・高耐熱性が特徴です。 

母材 | 強化材 | 製法 | 用途  
---|---|---|---  
Al合金 | SiC粒子(15-20 vol%) | 粉末冶金、溶湯攪拌 | 自動車エンジン部品  
Al合金 | Al₂O₃粒子 | スプレー成形 | ブレーキディスク  
Ti合金 | TiB繊維 | 反応合成 | 航空機構造材  
Cu合金 | グラファイト粒子 | 粉末冶金 | 電気接点、軸受  
  
### 3.1.3 CMC (Ceramic Matrix Composites)

セラミックス母材に繊維や粒子を複合化し、脆性を改善した材料です。 

材料系 | 使用温度 | 特徴 | 用途  
---|---|---|---  
SiC/SiC | ~1400°C | 高靱性、耐酸化性 | ジェットエンジンノズル  
C/SiC | ~1600°C | 軽量、高耐熱 | 航空機ブレーキ  
Al₂O₃/Al₂O₃ | ~1200°C | 高硬度、耐摩耗 | 切削工具  
  
## 3.2 粒子強化の力学モデル

### 3.2.1 弾性率の予測

球状粒子による強化では、Hashin-Shtrikman の上限・下限モデルが よく用いられます。等方性材料の場合： 

#### 体積弾性率 (Bulk modulus)

$$K_c = K_m + \frac{V_p}{(K_p - K_m)^{-1} + 3(1-V_p)/(3K_m + 4G_m)}$$ 

#### せん断弾性率 (Shear modulus)

$$G_c = G_m + \frac{V_p}{(G_p - G_m)^{-1} + 6(K_m + 2G_m)(1-V_p)/(5G_m(3K_m + 4G_m))}$$ 

ヤング率とポアソン比は以下から計算：

$$E_c = \frac{9K_c G_c}{3K_c + G_c}, \quad \nu_c = \frac{3K_c - 2G_c}{2(3K_c + G_c)}$$ 

#### 例題 3.1: SiC/Al 複合材料の弾性率計算
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    def hashin_shtrikman_moduli(K_m, G_m, K_p, G_p, V_p):
        """
        Hashin-Shtrikman モデルによる複合材料の弾性率計算
    
        Parameters:
        -----------
        K_m, G_m : float
            母材の体積弾性率、せん断弾性率 [GPa]
        K_p, G_p : float
            粒子の体積弾性率、せん断弾性率 [GPa]
        V_p : float or array
            粒子体積分率
    
        Returns:
        --------
        E_c, nu_c : float or array
            複合材料のヤング率、ポアソン比
        """
        # 体積弾性率
        K_c = K_m + V_p / (1/(K_p - K_m) + 3*(1 - V_p)/(3*K_m + 4*G_m))
    
        # せん断弾性率
        G_c = G_m + V_p / (1/(G_p - G_m) + 6*(K_m + 2*G_m)*(1 - V_p)/(5*G_m*(3*K_m + 4*G_m)))
    
        # ヤング率とポアソン比
        E_c = 9 * K_c * G_c / (3 * K_c + G_c)
        nu_c = (3 * K_c - 2 * G_c) / (2 * (3 * K_c + G_c))
    
        return E_c, nu_c
    
    def E_nu_to_K_G(E, nu):
        """ヤング率・ポアソン比から体積・せん断弾性率への変換"""
        K = E / (3 * (1 - 2 * nu))
        G = E / (2 * (1 + nu))
        return K, G
    
    # Al合金母材の特性
    E_m = 70.0   # GPa
    nu_m = 0.33
    K_m, G_m = E_nu_to_K_G(E_m, nu_m)
    
    # SiC粒子の特性
    E_p = 450.0  # GPa
    nu_p = 0.17
    K_p, G_p = E_nu_to_K_G(E_p, nu_p)
    
    # 体積分率範囲
    V_p_range = np.linspace(0, 0.5, 100)
    
    # 弾性率計算
    E_c, nu_c = hashin_shtrikman_moduli(K_m, G_m, K_p, G_p, V_p_range)
    
    # 混合則(上限・下限)との比較
    E_voigt = E_m * (1 - V_p_range) + E_p * V_p_range  # 上限
    E_reuss = 1 / ((1 - V_p_range)/E_m + V_p_range/E_p)  # 下限
    
    # 可視化
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # ヤング率
    ax1.plot(V_p_range, E_c, 'b-', linewidth=2, label='Hashin-Shtrikman')
    ax1.plot(V_p_range, E_voigt, 'r--', linewidth=1.5, label='Voigt (上限)')
    ax1.plot(V_p_range, E_reuss, 'g--', linewidth=1.5, label='Reuss (下限)')
    ax1.fill_between(V_p_range, E_reuss, E_voigt, alpha=0.2, color='gray',
                      label='混合則の範囲')
    ax1.set_xlabel('SiC 体積分率')
    ax1.set_ylabel('ヤング率 [GPa]')
    ax1.set_title('SiC/Al 複合材料のヤング率')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # ポアソン比
    ax2.plot(V_p_range, nu_c, 'b-', linewidth=2, label='Hashin-Shtrikman')
    ax2.axhline(y=nu_m, color='r', linestyle='--', label=f'Al母材 ({nu_m:.2f})')
    ax2.axhline(y=nu_p, color='g', linestyle='--', label=f'SiC粒子 ({nu_p:.2f})')
    ax2.set_xlabel('SiC 体積分率')
    ax2.set_ylabel('ポアソン比')
    ax2.set_title('SiC/Al 複合材料のポアソン比')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('particle_composite_modulus.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 実用的な粒子分率での値
    V_p_practical = np.array([0.10, 0.15, 0.20, 0.25, 0.30])
    E_c_practical, nu_c_practical = hashin_shtrikman_moduli(K_m, G_m, K_p, G_p, V_p_practical)
    
    print("SiC/Al 複合材料の弾性特性:")
    print("="*60)
    print(f"{'V_p':>6} {'E_c [GPa]':>12} {'増加率[%]':>12} {'ポアソン比':>12}")
    print("-"*60)
    for vp, ec, nuc in zip(V_p_practical, E_c_practical, nu_c_practical):
        increase = (ec / E_m - 1) * 100
        print(f"{vp:6.2f} {ec:12.1f} {increase:12.1f} {nuc:12.3f}")

### 3.2.2 強度の予測

粒子強化複合材料の強度は、以下の因子の複合効果で決まります： 

  * **荷重分担効果:** 粒子が荷重を負担
  * **転位強化:** 粒子周辺の転位密度増加
  * **Orowanメカニズム:** 転位が粒子をバイパス
  * **熱膨張差:** 冷却時の残留応力

#### 例題 3.2: 粒子強化複合材料の降伏強度予測
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    def particle_strengthening(sigma_m, V_p, d_p, b, G_m):
        """
        粒子強化複合材料の降伏強度予測
    
        Parameters:
        -----------
        sigma_m : float
            母材の降伏強度 [MPa]
        V_p : float
            粒子体積分率
        d_p : float
            粒子直径 [μm]
        b : float
            バーガースベクトル [nm]
        G_m : float
            母材のせん断弾性率 [GPa]
    
        Returns:
        --------
        sigma_c : float
            複合材料の降伏強度 [MPa]
        """
        # 荷重分担項(簡易モデル)
        sigma_load = sigma_m * (1 + 0.5 * V_p)
    
        # Orowan 強化項
        # 粒子間距離の推定
        lambda_p = d_p * (np.sqrt(np.pi / (4 * V_p)) - 1)  # [μm]
    
        # Orowan 応力 [MPa]
        G_m_MPa = G_m * 1000  # GPa → MPa
        b_m = b * 1e-9  # nm → m
        lambda_p_m = lambda_p * 1e-6  # μm → m
    
        sigma_orowan = 0.4 * G_m_MPa * b_m / lambda_p_m / 1e6  # MPa
    
        # 総強度(簡易的な加算則)
        sigma_c = sigma_load + sigma_orowan
    
        return sigma_c
    
    # Al合金母材
    sigma_m = 100  # MPa (焼鈍材)
    b = 0.286      # nm (Alのバーガースベクトル)
    G_m = 26       # GPa
    
    # SiC粒子サイズの影響
    d_p_range = np.logspace(-1, 1.5, 50)  # 0.1-30 μm
    V_p_values = [0.10, 0.15, 0.20, 0.25]
    
    plt.figure(figsize=(10, 6))
    
    for V_p in V_p_values:
        sigma_c = []
        for d_p in d_p_range:
            s_c = particle_strengthening(sigma_m, V_p, d_p, b, G_m)
            sigma_c.append(s_c)
    
        plt.plot(d_p_range, sigma_c, linewidth=2, label=f'V_p = {V_p:.2f}')
    
    plt.xscale('log')
    plt.xlabel('粒子直径 [μm]')
    plt.ylabel('複合材料の降伏強度 [MPa]')
    plt.title('粒子サイズと降伏強度の関係 (SiC/Al)')
    plt.grid(True, alpha=0.3, which='both')
    plt.legend()
    plt.tight_layout()
    plt.savefig('particle_size_strengthening.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 最適粒子サイズの検討
    V_p_opt = 0.20
    d_p_test = np.array([0.5, 1.0, 2.0, 5.0, 10.0])
    
    print("粒子サイズと強化効果:")
    print("="*60)
    print(f"{'粒子直径 [μm]':>15} {'降伏強度 [MPa]':>18} {'強化率 [%]':>15}")
    print("-"*60)
    
    for d_p in d_p_test:
        sigma_c = particle_strengthening(sigma_m, V_p_opt, d_p, b, G_m)
        strengthening = (sigma_c / sigma_m - 1) * 100
        print(f"{d_p:15.1f} {sigma_c:18.1f} {strengthening:15.1f}")

## 3.3 Orowan メカニズム

### 3.3.1 転位と粒子の相互作用

Orowanメカニズムは、転位が粒子を切断できない場合に、 粒子間を**バイパス(迂回)** することで生じる強化機構です。 

$$\Delta\sigma_{\text{Orowan}} = \frac{0.4Gb}{\lambda}$$ 

ここで、\\(G\\): せん断弾性率、\\(b\\): バーガースベクトル、\\(\lambda\\): 粒子間距離 

粒子間距離は、粒子サイズと体積分率から推定できます：

$$\lambda \approx d_p \left(\sqrt{\frac{\pi}{4V_p}} - 1\right)$$ 
    
    
    ```mermaid
    flowchart TD
                                A[転位の移動] --> B{粒子との相互作用}
                                B --> C[粒子切断可能弱い界面]
                                B --> D[粒子切断不可硬質粒子]
    
                                C --> E[転位が粒子を切断強化効果小]
                                D --> F[Orowan バイパス]
    
                                F --> G[転位ループが粒子周囲に残留]
                                G --> H[後続転位の移動を阻害]
                                H --> I[強度上昇]
    
                                style A fill:#e1f5ff
                                style F fill:#ffe1e1
                                style I fill:#c8e6c9
                        
    3.3.2 最適粒子サイズ・分率の設計
    
                            Orowan強化を最大化するには、粒子間距離を最小化する必要があります。
                            ただし、以下のトレードオフが存在します：
                        
    
    粒子サイズ小 → 強化効果大 (λ減少) が、凝集しやすい
    体積分率大 → 強化効果大 (λ減少) が、延性低下
    粒子サイズ/分率の最適化が重要
    
    
    例題 3.3: Orowan 強化の最適設計
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    def orowan_stress(d_p, V_p, G_m, b):
        """
        Orowan 応力の計算
    
        Parameters:
        -----------
        d_p : float
            粒子直径 [μm]
        V_p : float
            粒子体積分率
        G_m : float
            母材せん断弾性率 [GPa]
        b : float
            バーガースベクトル [nm]
    
        Returns:
        --------
        sigma_orowan : float
            Orowan 応力 [MPa]
        """
        # 粒子間距離 [m]
        lambda_p = d_p * 1e-6 * (np.sqrt(np.pi / (4 * V_p)) - 1)
    
        # 粒子間距離が正の場合のみ計算
        if lambda_p <= 0:
            return 0
    
        # Orowan 応力 [MPa]
        G_m_Pa = G_m * 1e9  # GPa → Pa
        b_m = b * 1e-9      # nm → m
    
        sigma_orowan = 0.4 * G_m_Pa * b_m / lambda_p / 1e6  # MPa
    
        return sigma_orowan
    
    def ductility_reduction_factor(V_p):
        """
        延性低下係数の推定(経験的モデル)
    
        V_p が大きいほど延性は低下する
        """
        return np.exp(-3 * V_p)
    
    # Al合金パラメータ
    G_m = 26  # GPa
    b = 0.286  # nm
    
    # パラメータ範囲
    d_p_range = np.logspace(-1, 1.2, 40)  # 0.1-16 μm
    V_p_range = np.linspace(0.05, 0.40, 40)
    
    # メッシュグリッド作成
    D_p, V_p_grid = np.meshgrid(d_p_range, V_p_range)
    
    # Orowan 応力の計算
    sigma_orowan_grid = np.zeros_like(D_p)
    performance_index = np.zeros_like(D_p)
    
    for i in range(len(V_p_range)):
        for j in range(len(d_p_range)):
            sigma_o = orowan_stress(D_p[i,j], V_p_grid[i,j], G_m, b)
            sigma_orowan_grid[i,j] = sigma_o
    
            # 性能指数: 強度 × 延性係数
            ductility = ductility_reduction_factor(V_p_grid[i,j])
            performance_index[i,j] = sigma_o * ductility
    
    # 3D プロット
    fig = plt.figure(figsize=(16, 6))
    
    # Orowan 応力
    ax1 = fig.add_subplot(131, projection='3d')
    surf1 = ax1.plot_surface(np.log10(D_p), V_p_grid, sigma_orowan_grid,
                              cmap='viridis', alpha=0.8)
    ax1.set_xlabel('log₁₀(粒子直径) [μm]')
    ax1.set_ylabel('体積分率')
    ax1.set_zlabel('Orowan応力 [MPa]')
    ax1.set_title('Orowan 強化効果')
    fig.colorbar(surf1, ax=ax1, shrink=0.5)
    
    # 性能指数
    ax2 = fig.add_subplot(132, projection='3d')
    surf2 = ax2.plot_surface(np.log10(D_p), V_p_grid, performance_index,
                              cmap='plasma', alpha=0.8)
    ax2.set_xlabel('log₁₀(粒子直径) [μm]')
    ax2.set_ylabel('体積分率')
    ax2.set_zlabel('性能指数 [強度×延性]')
    ax2.set_title('総合性能指数')
    fig.colorbar(surf2, ax=ax2, shrink=0.5)
    
    # 等高線プロット
    ax3 = fig.add_subplot(133)
    contour = ax3.contourf(np.log10(D_p), V_p_grid, performance_index,
                            levels=20, cmap='plasma')
    ax3.set_xlabel('log₁₀(粒子直径) [μm]')
    ax3.set_ylabel('体積分率')
    ax3.set_title('性能指数の等高線')
    fig.colorbar(contour, ax=ax3)
    
    # 最適点を探す
    max_idx = np.unravel_index(np.argmax(performance_index), performance_index.shape)
    d_p_opt = D_p[max_idx]
    V_p_opt = V_p_grid[max_idx]
    sigma_opt = sigma_orowan_grid[max_idx]
    
    ax3.plot(np.log10(d_p_opt), V_p_opt, 'r*', markersize=15,
             label=f'最適点: d_p={d_p_opt:.2f} μm, V_p={V_p_opt:.2f}')
    ax3.legend()
    
    plt.tight_layout()
    plt.savefig('orowan_optimization.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Orowan 強化の最適設計:")
    print("="*60)
    print(f"最適粒子直径: {d_p_opt:.2f} μm")
    print(f"最適体積分率: {V_p_opt:.2f}")
    print(f"Orowan 応力: {sigma_opt:.1f} MPa")
    print(f"粒子間距離: {d_p_opt * (np.sqrt(np.pi/(4*V_p_opt)) - 1):.2f} μm")
    ```

## 3.4 積層複合材料

### 3.4.1 積層材の種類と特徴

異なる材料を層状に積層することで、各層の特性を活かした複合材料を設計できます。 

積層系 | 構成 | 特徴 | 用途  
---|---|---|---  
金属積層材 | Al/Ti, Cu/Al | 熱伝導性、軽量化 | 熱交換器、電子機器  
クラッド鋼 | ステンレス/炭素鋼 | 耐食性+強度 | 化学プラント  
傾斜機能材料 | セラミック→金属 | 熱応力緩和 | 遮熱コーティング  
電磁シールド材 | Cu/樹脂/Cu | EMI遮蔽 | 電子基板  
  
### 3.4.2 積層材の熱応力

熱膨張係数が異なる材料を積層すると、温度変化により界面に応力が発生します。 

$$\sigma_{\text{thermal}} = \frac{E_1 E_2 (\alpha_1 - \alpha_2) \Delta T}{E_1 t_2 + E_2 t_1}$$ 

ここで、\\(E_i\\): 各層の弾性率、\\(\alpha_i\\): 熱膨張係数、\\(t_i\\): 層厚さ、\\(\Delta T\\): 温度変化 

#### 例題 3.4: 積層材の熱応力解析
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    def thermal_stress_bilayer(E1, E2, alpha1, alpha2, t1, t2, delta_T):
        """
        二層積層材の熱応力計算
    
        Parameters:
        -----------
        E1, E2 : float
            各層のヤング率 [GPa]
        alpha1, alpha2 : float
            各層の熱膨張係数 [/°C]
        t1, t2 : float
            各層の厚さ [mm]
        delta_T : float
            温度変化 [°C]
    
        Returns:
        --------
        sigma1, sigma2 : float
            各層の熱応力 [MPa]
        """
        # 熱応力(簡易モデル)
        E1_GPa = E1 * 1000  # GPa → MPa
        E2_GPa = E2 * 1000
    
        sigma_thermal = (E1_GPa * E2_GPa * (alpha1 - alpha2) * delta_T /
                         (E1_GPa * t2 + E2_GPa * t1))
    
        # 層1は圧縮、層2は引張(α1 > α2 の場合)
        sigma1 = -sigma_thermal * t2 / t1
        sigma2 = sigma_thermal
    
        return sigma1, sigma2
    
    # Al/Ti 積層材
    E_Al = 70   # GPa
    E_Ti = 110  # GPa
    alpha_Al = 23e-6  # /°C
    alpha_Ti = 9e-6   # /°C
    
    # 層厚さ比を変えた場合
    t_total = 10  # mm (総厚さ)
    t1_ratio = np.linspace(0.1, 0.9, 50)
    t1 = t1_ratio * t_total
    t2 = (1 - t1_ratio) * t_total
    
    delta_T = -155  # °C (180°C → 25°C)
    
    sigma_Al = []
    sigma_Ti = []
    
    for t1_val, t2_val in zip(t1, t2):
        s_Al, s_Ti = thermal_stress_bilayer(E_Al, E_Ti, alpha_Al, alpha_Ti,
                                             t1_val, t2_val, delta_T)
        sigma_Al.append(s_Al)
        sigma_Ti.append(s_Ti)
    
    sigma_Al = np.array(sigma_Al)
    sigma_Ti = np.array(sigma_Ti)
    
    # 可視化
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # 熱応力
    ax1.plot(t1_ratio, sigma_Al, 'b-', linewidth=2, label='Al層応力')
    ax1.plot(t1_ratio, sigma_Ti, 'r-', linewidth=2, label='Ti層応力')
    ax1.axhline(y=0, color='k', linestyle='--', linewidth=0.5)
    ax1.set_xlabel('Al層厚さ比 (t_Al / t_total)')
    ax1.set_ylabel('熱応力 [MPa]')
    ax1.set_title(f'Al/Ti 積層材の熱応力 (ΔT = {delta_T}°C)')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 降伏強度との比較
    sigma_y_Al = 100  # MPa (焼鈍Al)
    sigma_y_Ti = 350  # MPa (純Ti)
    
    # 安全率
    SF_Al = np.abs(sigma_y_Al / sigma_Al)
    SF_Ti = np.abs(sigma_y_Ti / sigma_Ti)
    SF_min = np.minimum(SF_Al, SF_Ti)
    
    ax2.plot(t1_ratio, SF_Al, 'b-', linewidth=2, label='Al層安全率')
    ax2.plot(t1_ratio, SF_Ti, 'r-', linewidth=2, label='Ti層安全率')
    ax2.plot(t1_ratio, SF_min, 'k--', linewidth=2, label='最小安全率')
    ax2.axhline(y=1.0, color='g', linestyle=':', linewidth=1.5, label='安全限界')
    ax2.set_xlabel('Al層厚さ比 (t_Al / t_total)')
    ax2.set_ylabel('安全率')
    ax2.set_title('各層の安全率')
    ax2.set_ylim([0, 10])
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('laminate_thermal_stress.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 最適厚さ比(安全率最大)
    opt_idx = np.argmax(SF_min)
    t1_opt_ratio = t1_ratio[opt_idx]
    
    print("Al/Ti 積層材の熱応力解析:")
    print("="*60)
    print(f"温度変化: {delta_T}°C")
    print(f"最適Al層厚さ比: {t1_opt_ratio:.2f}")
    print(f"最小安全率: {SF_min[opt_idx]:.2f}")
    print(f"\n厚さ比 {t1_opt_ratio:.2f} での応力:")
    print(f"  Al層応力: {sigma_Al[opt_idx]:.1f} MPa")
    print(f"  Ti層応力: {sigma_Ti[opt_idx]:.1f} MPa")

### 3.4.3 傾斜機能材料 (FGM)

組成を連続的に変化させることで、熱応力を緩和した材料です。 代表例: ZrO₂(セラミック) → Ni(金属) の傾斜材料 

#### 例題 3.5: FGM の組成分布設計
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    def fgm_property_profile(z, n, prop_ceramic, prop_metal):
        """
        べき乗則に基づくFGMの特性分布
    
        Parameters:
        -----------
        z : array
            厚さ方向座標(0: セラミック側、1: 金属側)
        n : float
            傾斜指数(n=1: 線形、n>1: セラミック側に偏在)
        prop_ceramic, prop_metal : float
            セラミックと金属の特性値
    
        Returns:
        --------
        prop : array
            位置 z での特性値
        """
        V_metal = z**n
        prop = prop_ceramic * (1 - V_metal) + prop_metal * V_metal
        return prop
    
    # ZrO2/Ni FGM
    E_ZrO2 = 200   # GPa
    E_Ni = 210     # GPa
    alpha_ZrO2 = 10e-6   # /°C
    alpha_Ni = 13e-6     # /°C
    
    # 厚さ方向座標
    z = np.linspace(0, 1, 100)
    
    # 傾斜指数の影響
    n_values = [0.5, 1.0, 2.0, 5.0]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    for n in n_values:
        E_profile = fgm_property_profile(z, n, E_ZrO2, E_Ni)
        alpha_profile = fgm_property_profile(z, n, alpha_ZrO2, alpha_Ni)
    
        ax1.plot(z, E_profile, linewidth=2, label=f'n = {n}')
        ax2.plot(z, alpha_profile * 1e6, linewidth=2, label=f'n = {n}')
    
    ax1.set_xlabel('厚さ方向座標 z (0: ZrO₂, 1: Ni)')
    ax1.set_ylabel('ヤング率 [GPa]')
    ax1.set_title('FGM のヤング率分布')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    ax2.set_xlabel('厚さ方向座標 z (0: ZrO₂, 1: Ni)')
    ax2.set_ylabel('熱膨張係数 [×10⁻⁶ /°C]')
    ax2.set_title('FGM の熱膨張係数分布')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('fgm_property_profile.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("傾斜機能材料(FGM)の特性分布:")
    print("="*60)
    print(f"{'傾斜指数 n':>12} {'中央部 E [GPa]':>18} {'中央部 α [10⁻⁶/°C]':>25}")
    print("-"*60)
    
    for n in n_values:
        E_mid = fgm_property_profile(0.5, n, E_ZrO2, E_Ni)
        alpha_mid = fgm_property_profile(0.5, n, alpha_ZrO2, alpha_Ni)
        print(f"{n:12.1f} {E_mid:18.1f} {alpha_mid*1e6:25.2f}")

## 3.5 まとめ

本章では、粒子強化と積層複合材料について学びました：

  * 粒子強化複合材料の分類(分散強化、粒子強化、充填材)
  * MMC/CMC の種類と製造法
  * Hashin-Shtrikman モデルによる弾性率予測
  * Orowan メカニズムと最適粒子設計
  * 積層材の熱応力と傾斜機能材料(FGM)

次章では、複合材料の機械的評価法(引張試験、曲げ試験、衝撃試験)と 非破壊検査(超音波、X線CT、サーモグラフィ)について学びます。 

## 演習問題

### 基礎レベル

#### 問題 3.1: Hashin-Shtrikman モデル

Al₂O₃粒子(E=380 GPa, ν=0.23)を20 vol%含むAl合金(E=70 GPa, ν=0.33)複合材料の ヤング率をHashin-Shtrikmanモデルで計算せよ。 

#### 問題 3.2: 粒子間距離の計算

直径 2 μm の SiC粒子を15 vol%含む複合材料の平均粒子間距離を求めよ。 

#### 問題 3.3: 熱応力の計算

Cu(E=120 GPa, α=17×10⁻⁶ /°C)とAl(E=70 GPa, α=23×10⁻⁶ /°C)の 二層積層材(各層1 mm)に100°Cの温度低下が生じた場合の熱応力を求めよ。 

### 応用レベル

#### 問題 3.4: Orowan 強化の最適化

Al合金(G=26 GPa, b=0.286 nm)に対し、目標降伏強度200 MPaを達成する SiC粒子の最適サイズと体積分率の組み合わせを求めよ。 (母材降伏強度: 100 MPa) 

#### 問題 3.5: MMC の設計

自動車エンジンピストン用のSiC/Al複合材料を設計せよ。 要求特性: ヤング率 ≥ 100 GPa、密度 ≤ 2.9 g/cm³ 

#### 問題 3.6: 積層材の最適化

Al/Ti積層材(総厚さ5 mm)の層厚さ比を最適化し、 200°Cの温度変化に対する最小安全率を最大化せよ。 

#### 問題 3.7: プログラミング課題

粒子強化複合材料の特性予測プログラムを作成せよ： 

  * Hashin-Shtrikman モデルで弾性率計算
  * Orowan モデルで強度計算
  * 粒子サイズ・分率に対する等高線プロット

### 発展レベル

#### 問題 3.8: 多目的最適化

SiC/Al複合材料について、以下を同時に最適化せよ： 

  * 目的1: 比強度(強度/密度)を最大化
  * 目的2: コストを最小化
  * 制約: ヤング率 ≥ 90 GPa

Pareto最適解をプロットせよ。

#### 問題 3.9: FGM の熱応力解析

ZrO₂/Ni傾斜機能材料(厚さ10 mm)について、有限要素法を用いて 温度分布と熱応力分布を計算せよ。 (表面温度: ZrO₂側 1200°C、Ni側 400°C) 

#### 問題 3.10: ナノ粒子分散強化

ナノサイズのAl₂O₃粒子(直径10-100 nm)によるODS合金の 強化機構を解析せよ。粒子サイズが10 nm以下になると Orowanメカニズムから転位切断メカニズムへ遷移する 臨界サイズを求めよ。 

## 参考文献

  1. Chawla, N. and Chawla, K. K., "Metal Matrix Composites", 2nd ed., Springer, 2013, pp. 89-156, 234-278
  2. Clyne, T. W. and Withers, P. J., "An Introduction to Metal Matrix Composites", Cambridge University Press, 1993, pp. 67-112
  3. Kainer, K. U., "Metal Matrix Composites: Custom-made Materials for Automotive and Aerospace Engineering", Wiley-VCH, 2006, pp. 45-89
  4. Courtney, T. H., "Mechanical Behavior of Materials", 2nd ed., Waveland Press, 2005, pp. 389-445
  5. Hashin, Z. and Shtrikman, S., "A Variational Approach to the Theory of the Elastic Behaviour of Multiphase Materials", Journal of the Mechanics and Physics of Solids, Vol. 11, 1963, pp. 127-140
  6. Koizumi, M., "FGM Activities in Japan", Composites Part B, Vol. 28, 1997, pp. 1-4
  7. Suresh, S. and Mortensen, A., "Fundamentals of Functionally Graded Materials", IOM Communications, 1998, pp. 23-67, 134-189
  8. Naebe, M. and Shirvanimoghaddam, K., "Functionally Graded Materials: A Review of Fabrication and Properties", Applied Materials Today, Vol. 5, 2016, pp. 223-245

### 免責事項

  * 本コンテンツは教育・研究・情報提供のみを目的としており、専門的な助言(法律・会計・技術的保証など)を提供するものではありません。
  * 本コンテンツおよび付随するCode examplesは「現状有姿(AS IS)」で提供され、明示または黙示を問わず、商品性、特定目的適合性、権利非侵害、正確性・完全性、動作・安全性等いかなる保証もしません。
  * 外部リンク、第三者が提供するデータ・ツール・ライブラリ等の内容・可用性・安全性について、作成者および東北大学は一切の責任を負いません。
  * 本コンテンツの利用・実行・解釈により直接的・間接的・付随的・特別・結果的・懲罰的損害が生じた場合でも、適用法で許容される最大限の範囲で、作成者および東北大学は責任を負いません。
  * 本コンテンツの内容は、予告なく変更・更新・提供停止されることがあります。
  * 本コンテンツの著作権・ライセンスは明記された条件(例: CC BY 4.0)に従います。当該ライセンスは通常、無保証条項を含みます。
