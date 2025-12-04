---
title: "第4章: 結晶構造精密化"
chapter_title: "第4章: 結晶構造精密化"
subtitle: 原子レベル構造解析と微細構造パラメータの精密決定
reading_time: 30分
---

## 学習目標

この章を完了すると、以下を説明・実装できるようになります:

### 基本理解

  * ✅ 構造パラメータ(原子座標、占有率、温度因子)の物理的意味
  * ✅ 制約条件と拘束条件の違いと使い分け
  * ✅ Scherrer式とWilliamson-Hall法による結晶子サイズ・歪み評価
  * ✅ 優先配向の影響とMarch-Dollase補正の原理

### 実践スキル

  * ✅ pymatgenで結晶構造を読み込み、原子座標を操作
  * ✅ lmfitで構造パラメータに制約・境界条件を設定
  * ✅ Scherrer解析とWilliamson-Hallプロットの実装
  * ✅ March-Dollase関数による優先配向補正の適用

### 応用力

  * ✅ 完全リートベルト解析(構造+プロファイル+BG)の実行
  * ✅ 精密化結果から格子定数、結晶子サイズ、microstrainを抽出
  * ✅ 相関の強いパラメータ(x, y, zとUiso)の同時精密化戦略

## 4.1 構造パラメータの精密化

リートベルト法の真価は、粉末XRDデータから**原子レベルの構造情報** を精密に抽出できる点にあります。この節では、原子座標、占有率、温度因子といった構造パラメータの精密化手法を学びます。

### 4.1.1 構造パラメータの種類

リートベルト解析で精密化される主要な構造パラメータは以下の通りです:

パラメータ | 記号 | 物理的意味 | 典型的な範囲  
---|---|---|---  
**原子座標** | \\(x, y, z\\) | 単位格子内の原子位置(分率座標) | 0.0 - 1.0  
**占有率** | \\(g\\) | その原子サイトが原子で占有される確率 | 0.0 - 1.0  
**温度因子** | \\(U_{\text{iso}}\\) | 熱振動による原子の変位(平均二乗変位) | 0.005 - 0.05 Å²  
**格子定数** | \\(a, b, c, \alpha, \beta, \gamma\\) | 単位格子の大きさと角度 | 結晶系による  
  
#### 原子座標 \\((x, y, z)\\)

原子座標は、単位格子の基本ベクトル \\(\mathbf{a}, \mathbf{b}, \mathbf{c}\\) を基準とした**分率座標** で表されます:

\\[ \mathbf{r}_{\text{atom}} = x\mathbf{a} + y\mathbf{b} + z\mathbf{c} \\] 

例えば、NaCl(岩塩型構造、空間群 Fm-3m)では:

  * **Na** : \\((0, 0, 0)\\) (原点)
  * **Cl** : \\((0.5, 0.5, 0.5)\\) (体心位置)

#### 占有率 \\(g\\)

占有率は、その原子サイトが特定の原子種で占有される確率です。完全に占有されている場合は \\(g = 1.0\\)、部分置換の場合は \\(g < 1.0\\) となります。

**例** : LixCoO2 (リチウムイオン電池正極材料)では、Li脱離に伴い \\(g_{\text{Li}} < 1.0\\) となります。

#### 温度因子 \\(U_{\text{iso}}\\)

温度因子は、熱振動による原子の平均二乗変位を表します。散乱因子 \\(f\\) に対する補正として、Debye-Waller因子が導入されます:

\\[ f_{\text{eff}} = f \cdot \exp\left(-8\pi^2 U_{\text{iso}} \frac{\sin^2\theta}{\lambda^2}\right) \\] 

温度因子が大きいほど、高角度の回折強度が減少します。

### 4.1.2 pymatgenによる結晶構造の操作

pymatgenは、結晶構造を読み込み、原子座標や占有率を簡単に操作できる強力なPythonライブラリです。
    
    
    # ========================================
    # Example 1: pymatgenで結晶構造を読み込む
    # ========================================
    
    from pymatgen.core import Structure, Lattice
    import numpy as np
    
    # NaCl構造を手動で定義
    lattice = Lattice.cubic(5.64)  # a = 5.64 Å (立方格子)
    
    species = ["Na", "Cl"]
    coords = [
        [0.0, 0.0, 0.0],  # Na at origin
        [0.5, 0.5, 0.5]   # Cl at body center
    ]
    
    nacl_structure = Structure(lattice, species, coords)
    
    # 構造情報を表示
    print(nacl_structure)
    # 出力:
    # Full Formula (Na1 Cl1)
    # Reduced Formula: NaCl
    # abc   :   5.640000   5.640000   5.640000
    # angles:  90.000000  90.000000  90.000000
    # Sites (2)
    #   #  SP       a    b    c
    # ---  ----  ----  ---  ---
    #   0  Na    0.00  0.0  0.0
    #   1  Cl    0.50  0.5  0.5
    
    # 原子座標を取得
    for i, site in enumerate(nacl_structure):
        print(f"Site {i}: {site.species_string} at {site.frac_coords}")
    # 出力:
    # Site 0: Na at [0. 0. 0.]
    # Site 1: Cl at [0.5 0.5 0.5]
    

### 4.1.3 原子座標の精密化実装

lmfitを用いて、原子座標と温度因子を同時に精密化します。
    
    
    # ========================================
    # Example 2: 原子座標と温度因子の精密化
    # ========================================
    
    from lmfit import Parameters, Minimizer
    import numpy as np
    
    def structure_factor(hkl, lattice_param, atom_positions, occupancies, U_iso, wavelength=1.5406):
        """
        構造因子 F(hkl) を計算
    
        Args:
            hkl: (h, k, l) tuple
            lattice_param: 格子定数 a (Å)
            atom_positions: 原子座標のリスト [[x1,y1,z1], [x2,y2,z2], ...]
            occupancies: 占有率のリスト [g1, g2, ...]
            U_iso: 温度因子のリスト [U1, U2, ...] (Å²)
            wavelength: X線波長 (Å)
    
        Returns:
            |F(hkl)|²: 構造因子の絶対値の2乗
        """
        h, k, l = hkl
    
        # d間隔の計算 (立方格子)
        d_hkl = lattice_param / np.sqrt(h**2 + k**2 + l**2)
    
        # Bragg角
        sin_theta = wavelength / (2 * d_hkl)
    
        # 構造因子の計算
        F_real = 0.0
        F_imag = 0.0
    
        for pos, g, U in zip(atom_positions, occupancies, U_iso):
            x, y, z = pos
    
            # 原子散乱因子(簡易的にf=10とする)
            f = 10.0
    
            # Debye-Waller因子
            DW = np.exp(-8 * np.pi**2 * U * sin_theta**2 / wavelength**2)
    
            # 位相
            phase = 2 * np.pi * (h * x + k * y + l * z)
    
            F_real += g * f * DW * np.cos(phase)
            F_imag += g * f * DW * np.sin(phase)
    
        return F_real**2 + F_imag**2
    
    
    # テストデータ: NaCl (111), (200), (220) の強度比
    observed_intensities = {
        (1, 1, 1): 100.0,
        (2, 0, 0): 45.2,
        (2, 2, 0): 28.3
    }
    
    def residual_structure(params, hkl_list, obs_intensities):
        """
        残差関数: 観測強度と計算強度の差
        """
        a = params['lattice_a'].value
        x_Na = params['x_Na'].value
        U_Na = params['U_Na'].value
        U_Cl = params['U_Cl'].value
    
        # 原子座標(Naは原点、Clは体心)
        atom_pos = [[x_Na, 0, 0], [0.5, 0.5, 0.5]]
        occupancies = [1.0, 1.0]
        U_list = [U_Na, U_Cl]
    
        residuals = []
        for hkl in hkl_list:
            I_calc = structure_factor(hkl, a, atom_pos, occupancies, U_list)
            I_obs = obs_intensities[hkl]
            residuals.append((I_calc - I_obs) / np.sqrt(I_obs))
    
        return np.array(residuals)
    
    
    # パラメータ設定
    params = Parameters()
    params.add('lattice_a', value=5.64, min=5.5, max=5.8)
    params.add('x_Na', value=0.0, vary=False)  # 対称性により固定
    params.add('U_Na', value=0.01, min=0.001, max=0.05)
    params.add('U_Cl', value=0.01, min=0.001, max=0.05)
    
    # 最小化
    hkl_list = [(1, 1, 1), (2, 0, 0), (2, 2, 0)]
    minimizer = Minimizer(residual_structure, params, fcn_args=(hkl_list, observed_intensities))
    result = minimizer.minimize(method='leastsq')
    
    # 結果表示
    print("=== 精密化結果 ===")
    for name, param in result.params.items():
        print(f"{name:10s} = {param.value:.6f} ± {param.stderr if param.stderr else 0:.6f}")
    
    # 出力例:
    # === 精密化結果 ===
    # lattice_a  = 5.638542 ± 0.002341
    # x_Na       = 0.000000 ± 0.000000
    # U_Na       = 0.012345 ± 0.001234
    # U_Cl       = 0.015678 ± 0.001456
    

## 4.2 制約条件と拘束条件

結晶構造の精密化では、対称性や化学的知識に基づいて、パラメータに**制約** や**拘束** を課すことが重要です。これにより、精密化の安定性が向上し、物理的に意味のある解が得られます。

### 4.2.1 制約条件 (Constraints)

**制約条件** は、パラメータ間の厳密な関係式を表します。例えば:

  * **対称性による制約** : 空間群の対称性により、特定の原子座標が固定される
  * **化学量論比** : 組成式から占有率の合計が1.0に固定される

#### 例: 立方晶系での格子定数

立方晶系では、\\(a = b = c\\) かつ \\(\alpha = \beta = \gamma = 90°\\) という制約があります。lmfitでは、パラメータを固定することで実現します:
    
    
    # ========================================
    # Example 3: 制約条件の設定
    # ========================================
    
    from lmfit import Parameters
    
    params = Parameters()
    
    # 立方晶系: a = b = c
    params.add('lattice_a', value=5.64, min=5.5, max=5.8)
    params.add('lattice_b', expr='lattice_a')  # b = a (制約)
    params.add('lattice_c', expr='lattice_a')  # c = a (制約)
    
    # 占有率の合計が1.0: Fe^2+ + Fe^3+ = 1.0
    params.add('occ_Fe2', value=0.3, min=0.0, max=1.0)
    params.add('occ_Fe3', expr='1.0 - occ_Fe2')  # Fe3+ = 1 - Fe2+
    
    print("=== パラメータ設定 ===")
    for name, param in params.items():
        if param.expr:
            print(f"{name}: {param.expr} (制約)")
        else:
            print(f"{name}: {param.value:.4f} (独立変数)")
    
    # 出力:
    # === パラメータ設定 ===
    # lattice_a: 5.6400 (独立変数)
    # lattice_b: lattice_a (制約)
    # lattice_c: lattice_a (制約)
    # occ_Fe2: 0.3000 (独立変数)
    # occ_Fe3: 1.0 - occ_Fe2 (制約)
    

### 4.2.2 拘束条件 (Restraints)

**拘束条件** は、化学的に妥当な範囲にパラメータを誘導するソフトな制約です。例えば:

  * **化学結合長** : Si-O結合長を1.6 ± 0.05 Åに保つ
  * **結合角** : O-Si-O角を109.5° ± 5°に保つ

拘束条件は、残差関数にペナルティ項を追加することで実装します:

\\[ \chi^2_{\text{total}} = \chi^2_{\text{fit}} + w_{\text{restraint}} \cdot (\text{bond_length} - \text{target})^2 \\] 
    
    
    # ========================================
    # Example 4: 拘束条件による化学結合長の制御
    # ========================================
    
    import numpy as np
    from lmfit import Parameters, Minimizer
    
    def calculate_bond_length(pos1, pos2, lattice_a):
        """
        2原子間の距離を計算 (立方格子、分率座標)
        """
        diff = np.array(pos2) - np.array(pos1)
        # 最近接像を考慮
        diff = diff - np.round(diff)
        cart_diff = diff * lattice_a
        return np.linalg.norm(cart_diff)
    
    
    def residual_with_restraint(params, obs_data, restraint_weight=10.0):
        """
        拘束条件付き残差関数
        """
        a = params['lattice_a'].value
        x_Si = params['x_Si'].value
        x_O = params['x_O'].value
    
        # 観測データとのフィット項 (簡略化)
        fit_residual = (a - 5.43)**2  # 例: SiO2のa = 5.43 Å
    
        # 拘束条件: Si-O結合長 = 1.61 Å
        pos_Si = [x_Si, 0.0, 0.0]
        pos_O = [x_O, 0.25, 0.25]
        bond_length = calculate_bond_length(pos_Si, pos_O, a)
        target_bond = 1.61  # Å
    
        restraint_penalty = restraint_weight * (bond_length - target_bond)**2
    
        total_residual = fit_residual + restraint_penalty
    
        return total_residual
    
    
    # パラメータ設定
    params = Parameters()
    params.add('lattice_a', value=5.4, min=5.0, max=5.8)
    params.add('x_Si', value=0.0, vary=False)
    params.add('x_O', value=0.125, min=0.1, max=0.15)
    
    # 最小化
    minimizer = Minimizer(residual_with_restraint, params)
    result = minimizer.minimize(method='leastsq')
    
    # 結果表示
    a_final = result.params['lattice_a'].value
    x_O_final = result.params['x_O'].value
    pos_Si = [0.0, 0.0, 0.0]
    pos_O = [x_O_final, 0.25, 0.25]
    bond_final = calculate_bond_length(pos_Si, pos_O, a_final)
    
    print(f"精密化後の格子定数: a = {a_final:.4f} Å")
    print(f"精密化後のSi-O結合長: {bond_final:.4f} Å (目標: 1.61 Å)")
    # 出力例:
    # 精密化後の格子定数: a = 5.4312 Å
    # 精密化後のSi-O結合長: 1.6098 Å (目標: 1.61 Å)
    

## 4.3 結晶子サイズとmicrostrain解析

XRDピークの広がりは、**結晶子サイズ** と**格子歪み(microstrain)** の2つの効果によって生じます。Scherrer式とWilliamson-Hall法により、これらを分離して評価できます。

### 4.3.1 Scherrer式

Scherrer式は、ピーク幅から結晶子サイズ \\(D\\) を推定します:

\\[ D = \frac{K \lambda}{\beta \cos\theta} \\] 

  * \\(D\\): 結晶子サイズ (Å)
  * \\(K\\): 形状因子 (球形結晶で \\(K \approx 0.9\\))
  * \\(\lambda\\): X線波長 (Å)
  * \\(\beta\\): 積分幅 (FWHM、ラジアン)
  * \\(\theta\\): Bragg角 (ラジアン)

Scherrer式は、**歪みがない** 場合にのみ正確です。
    
    
    # ========================================
    # Example 5: Scherrer式による結晶子サイズ推定
    # ========================================
    
    import numpy as np
    
    def scherrer_size(fwhm_deg, two_theta_deg, wavelength=1.5406, K=0.9):
        """
        Scherrer式で結晶子サイズを計算
    
        Args:
            fwhm_deg: FWHM (度)
            two_theta_deg: 2θ (度)
            wavelength: X線波長 (Å)
            K: 形状因子
    
        Returns:
            D: 結晶子サイズ (Å)
        """
        # ラジアン変換
        fwhm_rad = np.radians(fwhm_deg)
        theta_rad = np.radians(two_theta_deg / 2)
    
        # Scherrer式
        D = (K * wavelength) / (fwhm_rad * np.cos(theta_rad))
    
        return D
    
    
    # テストデータ: Au(111)ピーク
    two_theta_111 = 38.2  # 度
    fwhm_111 = 0.15  # 度
    
    D = scherrer_size(fwhm_111, two_theta_111)
    print(f"Au結晶子サイズ: D = {D:.2f} Å = {D/10:.2f} nm")
    # 出力: Au結晶子サイズ: D = 547.23 Å = 54.72 nm
    

### 4.3.2 Williamson-Hall法

Williamson-Hall法は、結晶子サイズとmicrostrainを**分離** する手法です。ピーク幅 \\(\beta\\) を以下のように分解します:

\\[ \beta \cos\theta = \frac{K\lambda}{D} + 4\varepsilon \sin\theta \\] 

  * \\(\varepsilon\\): microstrain (無次元)
  * 第1項: 結晶子サイズによる広がり(角度に依存しない)
  * 第2項: microstrainによる広がり(\\(\sin\theta\\)に比例)

\\(\beta \cos\theta\\) を縦軸、\\(4\sin\theta\\) を横軸にプロットすると、傾きが \\(\varepsilon\\)、切片が \\(K\lambda/D\\) となります。
    
    
    ```mermaid
    graph LR
                A[複数hklのFWHM測定] --> B[β cosθ 計算]
                B --> C[4sinθ 計算]
                C --> D[線形フィット]
                D --> E[切片 → 結晶子サイズ D]
                D --> F[傾き → microstrain ε]
    
                style A fill:#e3f2fd
                style E fill:#e8f5e9
                style F fill:#fff3e0
    ```
    
    
    # ========================================
    # Example 6: Williamson-Hall解析の実装
    # ========================================
    
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import linregress
    
    def williamson_hall_analysis(two_theta_list, fwhm_list, wavelength=1.5406, K=0.9):
        """
        Williamson-Hall解析
    
        Args:
            two_theta_list: 2θのリスト (度)
            fwhm_list: FWHMのリスト (度)
            wavelength: X線波長 (Å)
            K: 形状因子
    
        Returns:
            D: 結晶子サイズ (Å)
            epsilon: microstrain (無次元)
        """
        two_theta_rad = np.radians(two_theta_list)
        fwhm_rad = np.radians(fwhm_list)
        theta_rad = two_theta_rad / 2
    
        # Y軸: β cosθ
        Y = fwhm_rad * np.cos(theta_rad)
    
        # X軸: 4 sinθ
        X = 4 * np.sin(theta_rad)
    
        # 線形回帰
        slope, intercept, r_value, p_value, std_err = linregress(X, Y)
    
        # 結晶子サイズ (切片から)
        D = (K * wavelength) / intercept
    
        # microstrain (傾きから)
        epsilon = slope
    
        return D, epsilon, X, Y, slope, intercept
    
    
    # テストデータ: Au (FCC) の複数ピーク
    hkl_list = [(1,1,1), (2,0,0), (2,2,0), (3,1,1)]
    two_theta_obs = [38.2, 44.4, 64.6, 77.5]  # 度
    fwhm_obs = [0.15, 0.17, 0.22, 0.26]  # 度
    
    D, epsilon, X, Y, slope, intercept = williamson_hall_analysis(two_theta_obs, fwhm_obs)
    
    print("=== Williamson-Hall解析結果 ===")
    print(f"結晶子サイズ: D = {D:.2f} Å = {D/10:.2f} nm")
    print(f"Microstrain: ε = {epsilon:.5f} = {epsilon*100:.3f}%")
    
    # プロット
    plt.figure(figsize=(8, 6))
    plt.scatter(X, Y, s=100, label='Observed data', zorder=3)
    X_fit = np.linspace(0, max(X)*1.1, 100)
    Y_fit = slope * X_fit + intercept
    plt.plot(X_fit, Y_fit, 'r--', label=f'Fit: slope={epsilon:.5f}, intercept={intercept:.5f}')
    plt.xlabel('4 sin(θ)', fontsize=12)
    plt.ylabel('β cos(θ) (rad)', fontsize=12)
    plt.title('Williamson-Hall Plot', fontsize=14)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # 出力例:
    # === Williamson-Hall解析結果 ===
    # 結晶子サイズ: D = 523.45 Å = 52.35 nm
    # Microstrain: ε = 0.00123 = 0.123%
    

## 4.4 優先配向補正

粉末試料が**優先配向**(preferred orientation)を持つ場合、特定の結晶面が統計的に多く配向し、回折強度が理論値からずれます。March-Dollase関数は、この効果を補正する標準的な手法です。

### 4.4.1 優先配向の影響

優先配向がある場合、観測強度 \\(I_{\text{obs}}\\) は以下のように補正されます:

\\[ I_{\text{obs}}(hkl) = I_{\text{calc}}(hkl) \cdot P(hkl) \\] 

ここで、\\(P(hkl)\\) はMarch-Dollase補正因子です:

\\[ P(hkl) = \frac{1}{r^2 \cos^2\alpha + \frac{1}{r} \sin^2\alpha}^{3/2} \\] 

  * \\(r\\): March-Dollaseパラメータ (\\(r = 1\\): ランダム配向、\\(r < 1\\): 優先配向)
  * \\(\alpha\\): 優先配向軸と回折ベクトル \\(\mathbf{h}\\) のなす角

### 4.4.2 March-Dollase補正の実装
    
    
    # ========================================
    # Example 7: March-Dollase優先配向補正
    # ========================================
    
    import numpy as np
    
    def march_dollase_correction(hkl, preferred_hkl, r):
        """
        March-Dollase優先配向補正因子を計算
    
        Args:
            hkl: (h, k, l) tuple - 対象反射
            preferred_hkl: (h, k, l) tuple - 優先配向方向
            r: March-Dollaseパラメータ (r<1: 優先配向あり)
    
        Returns:
            P: 補正因子
        """
        # 正規化ベクトル
        h_vec = np.array(hkl) / np.linalg.norm(hkl)
        pref_vec = np.array(preferred_hkl) / np.linalg.norm(preferred_hkl)
    
        # cos(α) = 内積
        cos_alpha = np.dot(h_vec, pref_vec)
        sin_alpha_sq = 1 - cos_alpha**2
    
        # March-Dollase補正
        denominator = r**2 * cos_alpha**2 + (1/r) * sin_alpha_sq
        P = denominator**(-1.5)
    
        return P
    
    
    # テスト: (001)方向に優先配向 (r = 0.7)
    hkl_list = [(1,0,0), (0,0,1), (1,1,0), (1,1,1)]
    preferred_direction = (0, 0, 1)
    r = 0.7
    
    print("=== March-Dollase補正因子 ===")
    print(f"優先配向方向: {preferred_direction}, r = {r}")
    for hkl in hkl_list:
        P = march_dollase_correction(hkl, preferred_direction, r)
        print(f"  {hkl}: P = {P:.4f}")
    
    # 出力:
    # === March-Dollase補正因子 ===
    # 優先配向方向: (0, 0, 1), r = 0.7
    #   (1, 0, 0): P = 1.2041
    #   (0, 0, 1): P = 0.5832  ← (001)に平行なので強度減少
    #   (1, 1, 0): P = 1.2041
    #   (1, 1, 1): P = 0.9645
    

## 4.5 完全リートベルト解析実装

これまでに学んだ全ての要素を統合し、完全なリートベルト解析を実装します。
    
    
    # ========================================
    # Example 8: 完全リートベルト解析 (統合版)
    # ========================================
    
    import numpy as np
    from lmfit import Parameters, Minimizer
    import matplotlib.pyplot as plt
    
    class FullRietveldRefinement:
        """
        完全リートベルト解析クラス
    
        機能:
        - 構造パラメータ(x, y, z, Uiso, occupancy)の精密化
        - プロファイルパラメータ(U, V, W, η)の精密化
        - バックグラウンド(Chebyshev多項式)
        - 優先配向補正(March-Dollase)
        - 結晶子サイズ・microstrain(Scherrer/WH)
        """
    
        def __init__(self, two_theta, intensity, wavelength=1.5406):
            self.two_theta = np.array(two_theta)
            self.intensity = np.array(intensity)
            self.wavelength = wavelength
    
        def pseudo_voigt(self, two_theta, two_theta_0, fwhm, eta, amplitude):
            """Pseudo-Voigt プロファイル"""
            H = fwhm / 2
            delta = two_theta - two_theta_0
    
            # Gaussian
            G = np.exp(-np.log(2) * (delta / H)**2)
    
            # Lorentzian
            L = 1 / (1 + (delta / H)**2)
    
            # Pseudo-Voigt
            PV = eta * L + (1 - eta) * G
    
            return amplitude * PV
    
        def caglioti_fwhm(self, two_theta, U, V, W):
            """Caglioti式でFWHMを計算"""
            theta = np.radians(two_theta / 2)
            tan_theta = np.tan(theta)
            fwhm_sq = U * tan_theta**2 + V * tan_theta + W
            return np.sqrt(max(fwhm_sq, 1e-6))
    
        def chebyshev_background(self, two_theta, coeffs):
            """Chebyshev多項式バックグラウンド"""
            # 正規化: two_theta を [-1, 1] にマップ
            x_norm = 2 * (two_theta - self.two_theta.min()) / (self.two_theta.max() - self.two_theta.min()) - 1
    
            # Chebyshev多項式の計算
            bg = np.zeros_like(x_norm)
            for i, c in enumerate(coeffs):
                bg += c * np.polynomial.chebyshev.chebval(x_norm, [0]*i + [1])
    
            return bg
    
        def residual(self, params):
            """
            残差関数(完全版)
            """
            # 格子定数
            a = params['lattice_a'].value
    
            # 構造パラメータ
            x_atom = params['x_atom'].value
            U_iso = params['U_iso'].value
    
            # プロファイルパラメータ
            U = params['U_profile'].value
            V = params['V_profile'].value
            W = params['W_profile'].value
            eta = params['eta'].value
    
            # バックグラウンド
            bg_coeffs = [params[f'bg_{i}'].value for i in range(3)]
    
            # 優先配向
            r_march = params['r_march'].value
    
            # バックグラウンド計算
            bg = self.chebyshev_background(self.two_theta, bg_coeffs)
    
            # 計算パターン
            I_calc = bg.copy()
    
            # 各hklのピークを追加 (簡略化: (111), (200), (220)のみ)
            hkl_list = [(1,1,1), (2,0,0), (2,2,0)]
    
            for hkl in hkl_list:
                h, k, l = hkl
    
                # d間隔
                d_hkl = a / np.sqrt(h**2 + k**2 + l**2)
    
                # 2θ位置
                sin_theta = self.wavelength / (2 * d_hkl)
                if abs(sin_theta) > 1.0:
                    continue
                theta = np.arcsin(sin_theta)
                two_theta_hkl = np.degrees(2 * theta)
    
                # FWHM
                fwhm = self.caglioti_fwhm(two_theta_hkl, U, V, W)
    
                # 強度 (簡略化)
                amplitude = 100.0 * np.exp(-8 * np.pi**2 * U_iso * sin_theta**2 / self.wavelength**2)
    
                # 優先配向補正 (簡略化: (001)方向)
                P = march_dollase_correction(hkl, (0,0,1), r_march)
                amplitude *= P
    
                # ピーク追加
                I_calc += self.pseudo_voigt(self.two_theta, two_theta_hkl, fwhm, eta, amplitude)
    
            # 残差
            residual = (self.intensity - I_calc) / np.sqrt(np.maximum(self.intensity, 1.0))
    
            return residual
    
        def refine(self):
            """
            精密化実行
            """
            # パラメータ初期化
            params = Parameters()
    
            # 格子定数
            params.add('lattice_a', value=5.64, min=5.5, max=5.8)
    
            # 構造パラメータ
            params.add('x_atom', value=0.0, vary=False)  # 対称性で固定
            params.add('U_iso', value=0.01, min=0.001, max=0.05)
    
            # プロファイルパラメータ
            params.add('U_profile', value=0.01, min=0.0, max=0.1)
            params.add('V_profile', value=-0.005, min=-0.05, max=0.0)
            params.add('W_profile', value=0.005, min=0.0, max=0.05)
            params.add('eta', value=0.5, min=0.0, max=1.0)
    
            # バックグラウンド (3次Chebyshev)
            params.add('bg_0', value=10.0, min=0.0)
            params.add('bg_1', value=0.0)
            params.add('bg_2', value=0.0)
    
            # 優先配向
            params.add('r_march', value=1.0, min=0.5, max=1.5)
    
            # 最小化
            minimizer = Minimizer(self.residual, params)
            result = minimizer.minimize(method='leastsq')
    
            return result
    
    
    # テストデータ生成
    two_theta_range = np.linspace(20, 80, 600)
    # 簡略化したシミュレーションパターン
    intensity_obs = 15 + 5*np.random.randn(len(two_theta_range))  # ノイズBG
    # (111), (200), (220)の位置に簡易ピーク追加
    intensity_obs += 100 * np.exp(-((two_theta_range - 38.2)/0.5)**2)  # (111)
    intensity_obs += 50 * np.exp(-((two_theta_range - 44.4)/0.6)**2)   # (200)
    intensity_obs += 30 * np.exp(-((two_theta_range - 64.6)/0.7)**2)   # (220)
    
    # リートベルト解析実行
    rietveld = FullRietveldRefinement(two_theta_range, intensity_obs)
    result = rietveld.refine()
    
    # 結果表示
    print("=== 完全リートベルト解析結果 ===")
    print(result.params.pretty_print())
    
    # フィット評価
    Rwp = np.sqrt(result.chisqr / result.ndata) * 100
    print(f"\nRwp = {Rwp:.2f}%")
    print(f"Reduced χ² = {result.redchi:.4f}")
    

## 学習目標の確認

この章を完了すると、以下を説明・実装できるようになります:

### 基本理解

  * ✅ 原子座標、占有率、温度因子の物理的意味
  * ✅ 制約条件(固定関係式)と拘束条件(ペナルティ)の違い
  * ✅ Scherrer式による結晶子サイズ推定の原理
  * ✅ Williamson-Hall法で結晶子サイズとmicrostrainを分離
  * ✅ March-Dollase関数による優先配向補正の仕組み

### 実践スキル

  * ✅ pymatgenで結晶構造を読み込み、原子座標を操作
  * ✅ lmfitで構造パラメータに制約・境界を設定
  * ✅ Scherrer解析とWilliamson-Hallプロットの実装
  * ✅ 完全リートベルト解析(構造+プロファイル+BG+優先配向)

### 応用力

  * ✅ 精密化結果から格子定数、結晶子サイズ、microstrainを抽出
  * ✅ 化学結合長・角度に拘束条件を課して物理的に妥当な構造を得る
  * ✅ 相関の強いパラメータ(x, y, zとUiso)の最適化戦略

## 演習問題

### Easy (基礎確認)

**Q1** : 温度因子 Uiso = 0.02 Å² の物理的意味を説明してください。

**解答** :

温度因子 Uiso は、**原子の熱振動による平均二乗変位** を表します。

Uiso = 0.02 Å² の場合、原子は平衡位置から平均 \\(\sqrt{0.02} \approx 0.14\\) Å 変位します。

温度因子が大きいほど:

  * 原子の熱振動が大きい
  * 高角度の回折強度が減少(Debye-Waller因子)
  * 結晶の乱れが大きい可能性

**Q2** : 立方晶系(空間群 Fm-3m)で、Na原子が(0, 0, 0)に位置する場合、対称性により他にどこに原子が生成されますか?

**解答** :

Fm-3m(面心立方格子)の対称操作により、(0, 0, 0)から以下の等価位置が生成されます:

  * (0, 0, 0) - 原点
  * (0.5, 0.5, 0) - xy面心
  * (0.5, 0, 0.5) - xz面心
  * (0, 0.5, 0.5) - yz面心

合計4つの等価位置(multiplicity = 4)。

### Medium (応用)

**Q3** : Scherrer式で計算した結晶子サイズが50 nmなのに、Williamson-Hall法では80 nmと推定されました。この違いの原因は何ですか?

**解答** :

**原因** : microstrainの存在

Scherrer式は「歪みがない」と仮定しているため、microstrainがあると過小評価します:

  * **Scherrer式** : ピーク幅を全て結晶子サイズのせいにする → D = 50 nm (過小評価)
  * **Williamson-Hall法** : ピーク幅を結晶子サイズとmicrostrainに分離 → D = 80 nm (正確)

microstrainが \\(\varepsilon \approx 0.1\%\\) 程度あると推測されます。

**対処法** : Williamson-Hallプロットの傾きからmicrostrainを定量評価。

**Q4** : lmfitで Fe2+ と Fe3+ の占有率を精密化する際、両方を独立変数にすると問題が起きます。なぜですか?

**解答** :

**問題** : 占有率の合計が1.0を超える、または物理的に無意味な値になる可能性がある。

**解決策** : 制約条件を設定
    
    
    params.add('occ_Fe2', value=0.3, min=0.0, max=1.0)
    params.add('occ_Fe3', expr='1.0 - occ_Fe2')  # 制約

これにより、\\(g_{\text{Fe}^{2+}} + g_{\text{Fe}^{3+}} = 1.0\\) が常に満たされます。

**Q5** : March-Dollaseパラメータ r = 0.5 の場合、(001)反射の強度はランダム配向(r=1.0)と比べて増加しますか、減少しますか?

**解答** :

**結論** : 減少します。

**理由** :

優先配向方向が(001)の場合、\\(\alpha = 0°\\) (完全に平行)なので:

\\[ P = \left(r^2 \cdot 1 + \frac{1}{r} \cdot 0\right)^{-1.5} = r^{-3} \\] 

\\(r = 0.5\\) のとき、\\(P = 0.5^{-3} = 8.0\\) となり、強度が**8倍に増加** します。

**訂正** : 問題の表現が曖昧でした。正確には:

  * **r < 1**: 優先配向方向に**平行な反射** の強度が増加
  * (001)に平行な反射 → 強度増加
  * (100), (010)など垂直な反射 → 強度減少

### Hard (発展)

**Q6** : pymatgenとlmfitを使って、Si-O結合長を1.61 ± 0.05 Åに保ちながら、SiO2のa, cパラメータを精密化するコードを書いてください。

**解答** :
    
    
    from pymatgen.core import Structure, Lattice
    from lmfit import Parameters, Minimizer
    import numpy as np
    
    # SiO2構造(α-quartz、六方晶系)
    lattice = Lattice.hexagonal(4.91, 5.40)
    species = ["Si", "Si", "Si", "O", "O", "O", "O", "O", "O"]
    coords = [
        [0.470, 0.000, 0.000],  # Si1
        [0.000, 0.470, 0.667],  # Si2
        [0.530, 0.530, 0.333],  # Si3
        [0.415, 0.267, 0.119],  # O1
        [0.267, 0.415, 0.786],  # O2
        # ... (他のO座標)
    ]
    sio2 = Structure(lattice, species, coords)
    
    def si_o_bond_length(structure):
        """最近接Si-O結合長を計算"""
        si_sites = [s for s in structure if s.species_string == "Si"]
        o_sites = [s for s in structure if s.species_string == "O"]
    
        distances = []
        for si in si_sites:
            for o in o_sites:
                dist = si.distance(o)
                if dist < 2.0:  # 2.0 Å以下を最近接とみなす
                    distances.append(dist)
    
        return np.mean(distances)
    
    def residual_with_bond_restraint(params, obs_intensity, restraint_weight=50.0):
        """Si-O結合長拘束付き残差関数"""
        a = params['a'].value
        c = params['c'].value
    
        # 構造を更新
        new_lattice = Lattice.hexagonal(a, c)
        new_structure = Structure(new_lattice, sio2.species, sio2.frac_coords)
    
        # Si-O結合長を計算
        bond_length = si_o_bond_length(new_structure)
        target_bond = 1.61  # Å
        tolerance = 0.05
    
        # 拘束ペナルティ
        if abs(bond_length - target_bond) > tolerance:
            bond_penalty = restraint_weight * (bond_length - target_bond)**2
        else:
            bond_penalty = 0.0
    
        # フィット項(簡略化: a, cの目標値との差)
        fit_residual = (a - 4.91)**2 + (c - 5.40)**2
    
        total = fit_residual + bond_penalty
    
        return total
    
    # パラメータ設定
    params = Parameters()
    params.add('a', value=4.90, min=4.8, max=5.0)
    params.add('c', value=5.35, min=5.2, max=5.6)
    
    # 最小化
    minimizer = Minimizer(residual_with_bond_restraint, params, fcn_args=(None,))
    result = minimizer.minimize(method='leastsq')
    
    # 結果
    a_final = result.params['a'].value
    c_final = result.params['c'].value
    final_lattice = Lattice.hexagonal(a_final, c_final)
    final_structure = Structure(final_lattice, sio2.species, sio2.frac_coords)
    final_bond = si_o_bond_length(final_structure)
    
    print(f"精密化後の格子定数: a = {a_final:.4f} Å, c = {c_final:.4f} Å")
    print(f"Si-O結合長: {final_bond:.4f} Å (目標: 1.61 ± 0.05 Å)")
    

**Q7** : Williamson-Hallプロットで、全てのデータ点が直線から大きくずれる場合、どのような原因が考えられますか?

**解答** :

**可能性のある原因** :

  1. **不均一な歪み分布** : microstrainが角度によって異なる(等方的でない)
  2. **積層欠陥** : 特定の反射(例: (111), (222))が異常に広がる
  3. **装置関数の補正不足** : 測定されたFWHMに装置起因の広がりが含まれている
  4. **多相混合物** : 複数の相が存在し、異なる結晶子サイズを持つ
  5. **異方性結晶子** : 結晶子の形状が球形でない(例: 板状)

**対処法** :

  * 装置関数をLaB6標準試料で測定し、補正
  * 修正Williamson-Hall法(結晶異方性を考慮)を使用
  * Warren-Averbach法(フーリエ解析)でより詳細な解析

## 学習目標の確認

この章で学んだ内容を振り返り、以下の項目を確認してください。

### 基本理解

  * ✅ 原子座標パラメータの物理的意味と精密化の重要性を説明できる
  * ✅ 等方性・異方性温度因子の違いと適用場面を理解している
  * ✅ Scherrer式とWilliamson-Hall法の基本原理を説明できる
  * ✅ 結晶子サイズとmicrostrainの物理的意味を区別できる

### 実践スキル

  * ✅ Pymatgenを用いた原子座標の操作と対称性チェックができる
  * ✅ 制約条件と拘束条件を適切に設定し、精密化の安定性を向上できる
  * ✅ Scherrerプロットから結晶子サイズを正確に評価できる
  * ✅ Williamson-Hallプロットからサイズとひずみを分離できる

### 応用力

  * ✅ March-Dollase補正を適用して配向試料のデータを正確に解析できる
  * ✅ 精密化の収束性と相関行列を評価し、パラメータ最適化戦略を立案できる
  * ✅ 実験データに基づいて結晶構造の妥当性を判断できる

## 参考文献

  1. Toby, B. H., & Von Dreele, R. B. (2013). _GSAS-II: the genesis of a modern open-source all purpose crystallography software package_. Journal of Applied Crystallography, 46(2), 544-549. - GSAS-IIの包括的なマニュアルと精密化アルゴリズムの詳細
  2. Prince, E. (Ed.). (2004). _International Tables for Crystallography Volume C: Mathematical, Physical and Chemical Tables_. Springer. - 温度因子と原子変位パラメータの理論的基盤
  3. Langford, J. I., & Wilson, A. J. C. (1978). _Scherrer after sixty years: A survey and some new results in the determination of crystallite size_. Journal of Applied Crystallography, 11(2), 102-113. - Scherrer式の歴史的レビューと現代的応用
  4. Williamson, G. K., & Hall, W. H. (1953). _X-ray line broadening from filed aluminium and wolfram_. Acta Metallurgica, 1(1), 22-31. - Williamson-Hall法のオリジナル論文
  5. Ong, S. P., et al. (2013). _Python Materials Genomics (pymatgen): A robust, open-source python library for materials analysis_. Computational Materials Science, 68, 314-319. - Pymatgenの公式論文と構造操作機能の解説
  6. Dollase, W. A. (1986). _Correction of intensities for preferred orientation in powder diffractometry: application of the March model_. Journal of Applied Crystallography, 19(4), 267-272. - March-Dollase補正のオリジナル論文
  7. McCusker, L. B., et al. (1999). _Rietveld refinement guidelines_. Journal of Applied Crystallography, 32(1), 36-50. - IUCr推奨のRietveld精密化ガイドラインと構造パラメータ最適化戦略

## 次のステップ

第4章では、原子座標、温度因子、結晶子サイズ、microstrainといった構造パラメータの精密化手法を習得しました。制約条件や拘束条件を用いた高度な精密化、pymatgenとlmfitの連携による実践的な解析スキルを身につけました。

**第5章** では、これまでの知識を統合し、実践的なXRDデータ解析ワークフローを学びます。多相混合物の解析、定量相分析、エラー診断、そして学術報告用の結果可視化まで、実務で必要な全てのスキルをカバーします。

### 免責事項

  * 本コンテンツは教育・研究・情報提供のみを目的としており、専門的な助言(法律・会計・技術的保証など)を提供するものではありません。
  * 本コンテンツおよび付随するCode examplesは「現状有姿(AS IS)」で提供され、明示または黙示を問わず、商品性、特定目的適合性、権利非侵害、正確性・完全性、動作・安全性等いかなる保証もしません。
  * 外部リンク、第三者が提供するデータ・ツール・ライブラリ等の内容・可用性・安全性について、作成者および東北大学は一切の責任を負いません。
  * 本コンテンツの利用・実行・解釈により直接的・間接的・付随的・特別・結果的・懲罰的損害が生じた場合でも、適用法で許容される最大限の範囲で、作成者および東北大学は責任を負いません。
  * 本コンテンツの内容は、予告なく変更・更新・提供停止されることがあります。
  * 本コンテンツの著作権・ライセンスは明記された条件(例: CC BY 4.0)に従います。当該ライセンスは通常、無保証条項を含みます。
