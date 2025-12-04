---
title: "第2章: Hall効果測定"
chapter_title: "第2章: Hall効果測定"
subtitle: Lorentz力からvan der Pauw Hall配置、多キャリア解析、温度依存性まで
reading_time: 45-55分
difficulty: 中級
code_examples: 7
---

Hall効果は、磁場中での電荷キャリアの偏向を利用して、キャリア密度、キャリア種別（電子/正孔）、移動度を決定する強力な手法です。この章では、Lorentz力に基づくHall効果の理論、Hall係数とキャリア密度の関係、van der Pauw Hall測定配置、多キャリア系の解析、温度依存性からのキャリア散乱機構の解明を学び、Pythonで実践的なHallデータ解析を行います。 

## 学習目標

この章を読むことで、以下を習得できます：

  * ✅ Lorentz力からHall効果の式を導出し、Hall電圧を計算できる
  * ✅ Hall係数 $R_H = 1/(ne)$ とキャリア密度の関係を理解できる
  * ✅ van der Pauw Hall測定配置と測定手順を説明できる
  * ✅ 移動度 $\mu = \sigma R_H$ を計算し、物理的意味を理解できる
  * ✅ 多キャリア系（two-band model）の解析ができる
  * ✅ 温度依存性からキャリア散乱機構を解析できる
  * ✅ Pythonで完全なHallデータ処理ワークフローを構築できる

## 2.1 Hall効果の理論

### 2.1.1 Lorentz力とHall電圧

**Hall効果** （1879年、Edwin Herbert Hall発見）は、電流が流れる導体に垂直な磁場をかけたとき、電流と磁場の両方に垂直な方向に電圧が発生する現象です。
    
    
    ```mermaid
    flowchart TD
        A[電流 Ix方向] --> B[磁場 Bz方向]
        B --> C[Lorentz力F = -e v × B]
        C --> D[キャリア偏向y方向]
        D --> E[Hall電圧 V_Hy方向に発生]
    
        style A fill:#99ccff,stroke:#0066cc,stroke-width:2px
        style B fill:#99ff99,stroke:#00cc00,stroke-width:2px
        style C fill:#ffeb99,stroke:#ffa500,stroke-width:2px
        style D fill:#ff9999,stroke:#ff0000,stroke-width:2px
        style E fill:#f093fb,stroke:#f5576c,stroke-width:2px,color:#fff
    ```

**物理的メカニズム** ：

  1. 導体に電流 $I$ を流すと、キャリア（電子）がドリフト速度 $v_x$ で x 方向に移動
  2. z 方向の磁場 $B_z$ がかかると、Lorentz力 $\vec{F} = q\vec{v} \times \vec{B}$ が働く
  3. 電子（$q = -e$）は y 方向に偏向し、一方の側面に蓄積
  4. 電荷蓄積により Hall電場 $E_y$ が発生
  5. 定常状態で、Lorentz力と電場力が釣り合う：$eE_y = ev_x B_z$

**Hall電圧の導出** ：

試料の幅を $w$、厚さを $t$、電流を $I$ とすると、電流密度は：

$$ j_x = \frac{I}{wt} $$ 

キャリア密度を $n$ とすると、電流密度とドリフト速度の関係は：

$$ j_x = nev_x \quad \Rightarrow \quad v_x = \frac{j_x}{ne} = \frac{I}{newt} $$ 

定常状態での力の釣り合いから：

$$ E_y = v_x B_z = \frac{IB_z}{newt} $$ 

**Hall電圧** $V_H$ は、試料幅 $w$ にわたる電場の積分：

$$ V_H = E_y \cdot w = \frac{IB_z}{net} $$ 

**Hall係数** $R_H$ は次のように定義されます：

$$ R_H = \frac{E_y}{j_x B_z} = \frac{V_H t}{IB_z} = \frac{1}{ne} $$ 

したがって、**キャリア密度** $n$ は Hall係数から直接求まります：

$$ n = \frac{1}{eR_H} $$ 

### 2.1.2 キャリア種別の判定

Hall係数の符号から、キャリアが電子か正孔かを判定できます：

キャリア種別 | Hall係数 $R_H$ | Hall電圧の符号 | 物理的解釈  
---|---|---|---  
**電子** （n型） | $R_H < 0$ | 負 | 電子が負電荷を運ぶ  
**正孔** （p型） | $R_H > 0$ | 正 | 正孔が正電荷を運ぶ  
  
#### コード例2-1: Hall係数とキャリア密度の計算
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    def calculate_hall_coefficient(V_H, I, B, t):
        """
        Hall係数を計算
    
        Parameters
        ----------
        V_H : float
            Hall電圧 [V]
        I : float
            電流 [A]
        B : float
            磁場 [T]
        t : float
            試料厚さ [m]
    
        Returns
        -------
        R_H : float
            Hall係数 [m^3/C]
        """
        R_H = V_H * t / (I * B)
        return R_H
    
    def calculate_carrier_density(R_H):
        """
        キャリア密度を計算
    
        Parameters
        ----------
        R_H : float
            Hall係数 [m^3/C]
    
        Returns
        -------
        n : float
            キャリア密度 [m^-3]
        carrier_type : str
            キャリア種別（'electron' or 'hole'）
        """
        e = 1.60218e-19  # 電荷素量 [C]
        n = 1 / (np.abs(R_H) * e)
        carrier_type = 'electron' if R_H < 0 else 'hole'
        return n, carrier_type
    
    # 測定例1：n型シリコン
    V_H1 = -2.5e-3  # Hall電圧 [V]（負：電子）
    I1 = 1e-3  # 電流 [A]
    B1 = 0.5  # 磁場 [T]
    t1 = 500e-9  # 厚さ [m] = 500 nm
    
    R_H1 = calculate_hall_coefficient(V_H1, I1, B1, t1)
    n1, type1 = calculate_carrier_density(R_H1)
    
    print("測定例1: n型シリコン")
    print(f"  Hall電圧: {V_H1 * 1e3:.2f} mV")
    print(f"  Hall係数: {R_H1:.3e} m³/C")
    print(f"  キャリア種別: {type1}")
    print(f"  キャリア密度: {n1:.3e} m⁻³ = {n1 / 1e6:.3e} cm⁻³")
    
    # 測定例2：p型ガリウムヒ素
    V_H2 = +3.8e-3  # Hall電圧 [V]（正：正孔）
    I2 = 1e-3  # 電流 [A]
    B2 = 0.5  # 磁場 [T]
    t2 = 300e-9  # 厚さ [m] = 300 nm
    
    R_H2 = calculate_hall_coefficient(V_H2, I2, B2, t2)
    n2, type2 = calculate_carrier_density(R_H2)
    
    print("\n測定例2: p型ガリウムヒ素")
    print(f"  Hall電圧: {V_H2 * 1e3:.2f} mV")
    print(f"  Hall係数: {R_H2:.3e} m³/C")
    print(f"  キャリア種別: {type2}")
    print(f"  キャリア密度: {n2:.3e} m⁻³ = {n2 / 1e6:.3e} cm⁻³")
    
    # キャリア密度依存性の可視化
    n_range = np.logspace(20, 28, 100)  # [m^-3]
    R_H_range = 1 / (n_range * 1.60218e-19)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # 左図：n vs R_H
    ax1.loglog(n_range / 1e6, np.abs(R_H_range), linewidth=2.5, color='#f093fb', label='|R_H| = 1/(ne)')
    ax1.scatter([n1 / 1e6], [np.abs(R_H1)], s=150, c='#f5576c', edgecolors='black', linewidth=2, zorder=5, label='n-Si (example 1)')
    ax1.scatter([n2 / 1e6], [np.abs(R_H2)], s=150, c='#ffa500', edgecolors='black', linewidth=2, zorder=5, label='p-GaAs (example 2)')
    ax1.set_xlabel('Carrier Density n [cm$^{-3}$]', fontsize=12)
    ax1.set_ylabel('|Hall Coefficient R$_H$| [m$^3$/C]', fontsize=12)
    ax1.set_title('Hall Coefficient vs Carrier Density', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(alpha=0.3, which='both')
    
    # 右図：Hall電圧の磁場依存性
    B_range = np.linspace(0, 1, 100)  # [T]
    V_H_n = R_H1 * I1 * B_range / t1 * 1e3  # n型 [mV]
    V_H_p = R_H2 * I2 * B_range / t2 * 1e3  # p型 [mV]
    
    ax2.plot(B_range, V_H_n, linewidth=2.5, color='#f5576c', label='n-type (electron, R_H < 0)')
    ax2.plot(B_range, V_H_p, linewidth=2.5, color='#ffa500', label='p-type (hole, R_H > 0)')
    ax2.axhline(0, color='black', linestyle='--', linewidth=1.5)
    ax2.set_xlabel('Magnetic Field B [T]', fontsize=12)
    ax2.set_ylabel('Hall Voltage V$_H$ [mV]', fontsize=12)
    ax2.set_title('Hall Voltage vs Magnetic Field', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    

## 2.2 移動度の決定

### 2.2.1 移動度とHall係数の関係

**移動度** $\mu$ は、電気伝導率 $\sigma$ とHall係数 $R_H$ から求まります：

$$ \mu = \sigma R_H $$ 

これは、$\sigma = ne\mu$ と $R_H = 1/(ne)$ から直接導かれます。

**物理的意味** ：

  * $\sigma$ は電気伝導のしやすさ（キャリア密度 $n$ と移動度 $\mu$ の両方に依存）
  * $R_H$ はキャリア密度 $n$ のみに依存
  * 両者を組み合わせることで、移動度 $\mu$ が分離して求まる

#### コード例2-2: 移動度の計算
    
    
    import numpy as np
    
    def calculate_mobility(sigma, R_H):
        """
        移動度を計算
    
        Parameters
        ----------
        sigma : float
            電気伝導率 [S/m]
        R_H : float
            Hall係数 [m^3/C]
    
        Returns
        -------
        mu : float
            移動度 [m^2/(V·s)]
        """
        mu = sigma * np.abs(R_H)
        return mu
    
    # 測定例：n型シリコン（前の例から）
    R_H = -2.5e-3  # [m^3/C]
    sigma = 1e4  # 電気伝導率 [S/m]（典型値）
    
    mu = calculate_mobility(sigma, R_H)
    
    e = 1.60218e-19  # [C]
    n = 1 / (np.abs(R_H) * e)
    
    print("n型シリコンの電気的特性:")
    print(f"  電気伝導率 σ = {sigma:.2e} S/m")
    print(f"  Hall係数 R_H = {R_H:.2e} m³/C")
    print(f"  キャリア密度 n = {n:.2e} m⁻³ = {n / 1e6:.2e} cm⁻³")
    print(f"  移動度 μ = {mu:.2e} m²/(V·s) = {mu * 1e4:.1f} cm²/(V·s)")
    print(f"\n検証: σ = neμ = {n * e * mu:.2e} S/m（一致）")
    
    # 材料比較
    materials = {
        'Si (bulk, n-type)': {'sigma': 1e4, 'R_H': -2.5e-3},
        'GaAs (bulk, n-type)': {'sigma': 1e5, 'R_H': -5e-3},
        'InSb (n-type)': {'sigma': 1e6, 'R_H': -1e-2},
        'Graphene': {'sigma': 1e5, 'R_H': -1e-4}
    }
    
    print("\n材料比較:")
    print(f"{'Material':<25} {'n [cm⁻³]':<15} {'μ [cm²/(V·s)]':<20}")
    print("-" * 60)
    
    for name, props in materials.items():
        n_mat = 1 / (np.abs(props['R_H']) * e)
        mu_mat = calculate_mobility(props['sigma'], props['R_H'])
        print(f"{name:<25} {n_mat / 1e6:.2e}      {mu_mat * 1e4:.1f}")
    

**出力解釈** ：

  * シリコンの移動度: 〜1000 cm²/(V·s)（典型値）
  * GaAsは高移動度材料（〜5000 cm²/(V·s)）
  * InSbは超高移動度（〜77,000 cm²/(V·s)）
  * Grapheneは極めて高い移動度（〜10,000-100,000 cm²/(V·s)）

## 2.3 van der Pauw Hall測定配置

### 2.3.1 8接点van der Pauw Hall配置

**van der Pauw Hall測定** は、任意形状の薄膜試料でHall効果を測定できる標準的な手法です。第1章で学んだシート抵抗測定と組み合わせて、1つの試料から $\sigma$、$R_H$、$n$、$\mu$ のすべてを決定できます。
    
    
    ```mermaid
    flowchart TD
        A[試料準備4隅に8接点配置] --> B[シート抵抗測定R_AB,CD, R_BC,DA]
        B --> C[シート抵抗 R_s 計算van der Pauw式]
        C --> D[Hall測定磁場 B をかける]
        D --> E[Hall電圧 V_H 測定電流 I で]
        E --> F[Hall係数 R_H 計算R_H = V_H t / IB]
        F --> G[キャリア密度 nn = 1/eR_H]
        F --> H[移動度 μμ = σR_H]
    
        style A fill:#99ccff,stroke:#0066cc,stroke-width:2px
        style C fill:#ffeb99,stroke:#ffa500,stroke-width:2px
        style F fill:#f093fb,stroke:#f5576c,stroke-width:2px,color:#fff
        style G fill:#99ff99,stroke:#00cc00,stroke-width:2px
        style H fill:#99ff99,stroke:#00cc00,stroke-width:2px
    ```

**測定手順** ：

  1. **シート抵抗測定** （磁場なし、B = 0）： 
     * 接点1→2に電流、3-4間で電圧測定 → $R_{12,34}$
     * 接点2→3に電流、4-1間で電圧測定 → $R_{23,41}$
     * van der Pauw式で $R_s$ を計算
  2. **Hall測定** （磁場 B をかける、例えば B = +0.5 T）： 
     * 接点1→3に電流 $I$、2-4間で電圧 $V_{24}^{+B}$ 測定
     * 磁場を反転（B = -0.5 T）、同じ電流で $V_{24}^{-B}$ 測定
     * Hall電圧：$V_H = \frac{1}{2}(V_{24}^{+B} - V_{24}^{-B})$
     * Hall係数：$R_H = \frac{V_H t}{IB}$
  3. **電気的特性の導出** ： 
     * 電気伝導率：$\sigma = \frac{1}{R_s t}$
     * キャリア密度：$n = \frac{1}{eR_H}$
     * 移動度：$\mu = \sigma R_H$

> **重要** ：磁場を反転して測定する理由は、オフセット電圧（熱電効果、不均一性による）をキャンセルするためです。Hall電圧は磁場に対して奇関数（$V_H(B) = -V_H(-B)$）ですが、オフセット電圧は偶関数なので、差分を取ることで純粋なHall電圧が得られます。 

#### コード例2-3: van der Pauw Hall測定のシミュレーション
    
    
    import numpy as np
    from scipy.optimize import fsolve
    
    def van_der_pauw_sheet_resistance(R1, R2):
        """van der Pauw式でシート抵抗を計算"""
        def equation(Rs):
            return np.exp(-np.pi * R1 / Rs) + np.exp(-np.pi * R2 / Rs) - 1
    
        R_initial = (R1 + R2) / 2 * np.pi / np.log(2)
        R_s = fsolve(equation, R_initial)[0]
        return R_s
    
    def complete_hall_analysis(R_AB_CD, R_BC_DA, V_24_pos_B, V_24_neg_B, I, B, t):
        """
        完全なvan der Pauw Hall解析
    
        Parameters
        ----------
        R_AB_CD, R_BC_DA : float
            van der Pauw抵抗 [Ω]
        V_24_pos_B, V_24_neg_B : float
            正負磁場でのHall電圧 [V]
        I : float
            電流 [A]
        B : float
            磁場の大きさ [T]
        t : float
            試料厚さ [m]
    
        Returns
        -------
        results : dict
            解析結果（R_s, sigma, R_H, n, mu）
        """
        e = 1.60218e-19  # [C]
    
        # 1. シート抵抗
        R_s = van_der_pauw_sheet_resistance(R_AB_CD, R_BC_DA)
    
        # 2. 電気伝導率
        sigma = 1 / (R_s * t)
    
        # 3. Hall電圧（オフセット除去）
        V_H = 0.5 * (V_24_pos_B - V_24_neg_B)
    
        # 4. Hall係数
        R_H = V_H * t / (I * B)
    
        # 5. キャリア密度
        n = 1 / (np.abs(R_H) * e)
        carrier_type = 'electron' if R_H < 0 else 'hole'
    
        # 6. 移動度
        mu = sigma * np.abs(R_H)
    
        results = {
            'R_s': R_s,
            'sigma': sigma,
            'rho': 1 / sigma,
            'V_H': V_H,
            'R_H': R_H,
            'n': n,
            'carrier_type': carrier_type,
            'mu': mu
        }
    
        return results
    
    # 測定データ例：n型シリコン薄膜
    R_AB_CD = 1000  # [Ω]
    R_BC_DA = 950   # [Ω]
    V_24_plus = -5.2e-3  # +B での電圧 [V]
    V_24_minus = +4.8e-3  # -B での電圧 [V]
    I = 100e-6  # 電流 [A] = 100 μA
    B = 0.5  # 磁場 [T]
    t = 200e-9  # 厚さ [m] = 200 nm
    
    results = complete_hall_analysis(R_AB_CD, R_BC_DA, V_24_plus, V_24_minus, I, B, t)
    
    print("van der Pauw Hall測定解析結果:")
    print("=" * 60)
    print(f"測定条件:")
    print(f"  R_AB,CD = {R_AB_CD:.1f} Ω")
    print(f"  R_BC,DA = {R_BC_DA:.1f} Ω")
    print(f"  V_24(+B) = {V_24_plus * 1e3:.2f} mV")
    print(f"  V_24(-B) = {V_24_minus * 1e3:.2f} mV")
    print(f"  電流 I = {I * 1e6:.1f} μA")
    print(f"  磁場 B = ±{B:.2f} T")
    print(f"  厚さ t = {t * 1e9:.0f} nm")
    print("\n解析結果:")
    print(f"  シート抵抗 R_s = {results['R_s']:.2f} Ω/sq")
    print(f"  電気伝導率 σ = {results['sigma']:.2e} S/m")
    print(f"  抵抗率 ρ = {results['rho']:.2e} Ω·m = {results['rho'] * 1e8:.2f} μΩ·cm")
    print(f"  Hall電圧 V_H = {results['V_H'] * 1e3:.2f} mV")
    print(f"  Hall係数 R_H = {results['R_H']:.2e} m³/C")
    print(f"  キャリア種別: {results['carrier_type']}")
    print(f"  キャリア密度 n = {results['n']:.2e} m⁻³ = {results['n'] / 1e6:.2e} cm⁻³")
    print(f"  移動度 μ = {results['mu']:.2e} m²/(V·s) = {results['mu'] * 1e4:.1f} cm²/(V·s)")
    print("\n検証:")
    print(f"  σ = neμ = {results['n'] * 1.60218e-19 * results['mu']:.2e} S/m")
    print(f"  （計算値 σ = {results['sigma']:.2e} S/m と一致）")
    

## 2.4 多キャリア解析（Two-Band Model）

### 2.4.1 2キャリア系の理論

半導体では、電子と正孔の両方が伝導に寄与する場合があります（例：狭バンドギャップ半導体、InSb、HgCdTeなど）。この場合、単純な1バンドモデルでは不十分で、**two-band model** が必要です。

**電気伝導率** （2キャリア）：

$$ \sigma = n_e e \mu_e + n_h e \mu_h $$ 

**Hall係数** （2キャリア）：

$$ R_H = \frac{n_h \mu_h^2 - n_e \mu_e^2}{e(n_h \mu_h + n_e \mu_e)^2} $$ 

ここで、$n_e$, $\mu_e$ は電子のキャリア密度と移動度、$n_h$, $\mu_h$ は正孔のキャリア密度と移動度です。

**物理的解釈** ：

  * $\mu_h \gg \mu_e$ の場合、正孔がHall効果を支配（$R_H > 0$）
  * $\mu_e \gg \mu_h$ の場合、電子がHall効果を支配（$R_H < 0$）
  * 移動度が同程度の場合、$R_H$ の符号はキャリア密度の比 $n_h/n_e$ に依存

#### コード例2-4: Two-Band Modelのフィッティング
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    from lmfit import Model
    
    def two_band_conductivity(n_e, mu_e, n_h, mu_h):
        """2バンドモデルの電気伝導率"""
        e = 1.60218e-19
        sigma = n_e * e * mu_e + n_h * e * mu_h
        return sigma
    
    def two_band_hall_coefficient(n_e, mu_e, n_h, mu_h):
        """2バンドモデルのHall係数"""
        e = 1.60218e-19
        numerator = n_h * mu_h**2 - n_e * mu_e**2
        denominator = (n_h * mu_h + n_e * mu_e)**2
        R_H = numerator / (e * denominator)
        return R_H
    
    # シミュレーション：InSb（電子・正孔共存系）
    T_range = np.linspace(200, 400, 50)  # 温度 [K]
    
    # 温度依存性（簡略化）
    n_e = 1e22 * np.exp(-0.1 / (8.617e-5 * T_range))  # 電子密度 [m^-3]
    n_h = 5e21 * np.exp(-0.08 / (8.617e-5 * T_range))  # 正孔密度 [m^-3]
    mu_e = 7e4 * (300 / T_range)**1.5 * 1e-4  # 電子移動度 [m^2/(V·s)]
    mu_h = 1e3 * (300 / T_range)**2.5 * 1e-4  # 正孔移動度 [m^2/(V·s)]
    
    sigma = two_band_conductivity(n_e, mu_e, n_h, mu_h)
    R_H = two_band_hall_coefficient(n_e, mu_e, n_h, mu_h)
    
    # プロット
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 左上：キャリア密度
    axes[0, 0].semilogy(T_range, n_e / 1e6, linewidth=2.5, label='Electron density n$_e$', color='#f5576c')
    axes[0, 0].semilogy(T_range, n_h / 1e6, linewidth=2.5, label='Hole density n$_h$', color='#ffa500')
    axes[0, 0].set_xlabel('Temperature T [K]', fontsize=12)
    axes[0, 0].set_ylabel('Carrier Density [cm$^{-3}$]', fontsize=12)
    axes[0, 0].set_title('Carrier Densities (Two-Band Model)', fontsize=13, fontweight='bold')
    axes[0, 0].legend(fontsize=11)
    axes[0, 0].grid(alpha=0.3)
    
    # 右上：移動度
    axes[0, 1].loglog(T_range, mu_e * 1e4, linewidth=2.5, label='Electron mobility μ$_e$', color='#f5576c')
    axes[0, 1].loglog(T_range, mu_h * 1e4, linewidth=2.5, label='Hole mobility μ$_h$', color='#ffa500')
    axes[0, 1].set_xlabel('Temperature T [K]', fontsize=12)
    axes[0, 1].set_ylabel('Mobility [cm$^2$/(V·s)]', fontsize=12)
    axes[0, 1].set_title('Mobilities (Temperature Dependence)', fontsize=13, fontweight='bold')
    axes[0, 1].legend(fontsize=11)
    axes[0, 1].grid(alpha=0.3, which='both')
    
    # 左下：電気伝導率
    axes[1, 0].semilogy(T_range, sigma, linewidth=2.5, color='#f093fb')
    axes[1, 0].set_xlabel('Temperature T [K]', fontsize=12)
    axes[1, 0].set_ylabel('Conductivity σ [S/m]', fontsize=12)
    axes[1, 0].set_title('Electrical Conductivity', fontsize=13, fontweight='bold')
    axes[1, 0].grid(alpha=0.3)
    
    # 右下：Hall係数
    axes[1, 1].plot(T_range, R_H, linewidth=2.5, color='#f093fb')
    axes[1, 1].axhline(0, color='black', linestyle='--', linewidth=1.5)
    axes[1, 1].set_xlabel('Temperature T [K]', fontsize=12)
    axes[1, 1].set_ylabel('Hall Coefficient R$_H$ [m$^3$/C]', fontsize=12)
    axes[1, 1].set_title('Hall Coefficient (Sign Change)', fontsize=13, fontweight='bold')
    axes[1, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Hall係数の符号反転温度を検出
    sign_change_idx = np.where(np.diff(np.sign(R_H)))[0]
    if len(sign_change_idx) > 0:
        T_sign_change = T_range[sign_change_idx[0]]
        print(f"\nHall係数の符号反転温度: {T_sign_change:.1f} K")
        print("  → 低温: R_H < 0（電子優勢）")
        print("  → 高温: R_H > 0（正孔優勢）")
    

## 2.5 温度依存性Hall測定

### 2.5.1 キャリア散乱機構の解析

移動度の温度依存性から、支配的なキャリア散乱機構を特定できます：

散乱機構 | 移動度の温度依存性 | 支配的な温度域 | 材料例  
---|---|---|---  
**音響フォノン散乱** | $\mu \propto T^{-3/2}$ | 室温以上 | Si, GaAs（高温）  
**イオン化不純物散乱** | $\mu \propto T^{3/2}$ | 低温（< 100 K） | ドープ半導体  
**光学フォノン散乱** | 複雑（温度に依存） | 高温 | 極性半導体（GaAs）  
**中性不純物散乱** | $\mu \approx$ 定数 | 低温 | 高ドープ材料  
  
**Matthiessenの法則** （移動度版）：

$$ \frac{1}{\mu_{\text{total}}} = \frac{1}{\mu_{\text{phonon}}} + \frac{1}{\mu_{\text{impurity}}} + \frac{1}{\mu_{\text{other}}} $$ 

#### コード例2-5: 温度依存性Hall測定の解析
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    from lmfit import Model
    
    # 音響フォノン散乱モデル
    def acoustic_phonon_mobility(T, mu0, T0=300):
        """μ ∝ T^(-3/2)"""
        return mu0 * (T0 / T)**(3/2)
    
    # イオン化不純物散乱モデル
    def ionized_impurity_mobility(T, mu1, T0=300):
        """μ ∝ T^(3/2)"""
        return mu1 * (T / T0)**(3/2)
    
    # Matthiessenの法則
    def combined_mobility(T, mu0, mu1, T0=300):
        """1/μ_total = 1/μ_phonon + 1/μ_impurity"""
        mu_phonon = acoustic_phonon_mobility(T, mu0, T0)
        mu_impurity = ionized_impurity_mobility(T, mu1, T0)
        mu_total = 1 / (1/mu_phonon + 1/mu_impurity)
        return mu_total
    
    # シミュレーションデータ生成
    T_range = np.linspace(50, 400, 30)  # [K]
    mu0_true = 8000  # フォノン散乱制限移動度（室温）[cm^2/(V·s)]
    mu1_true = 2000  # 不純物散乱制限移動度（室温）[cm^2/(V·s)]
    
    mu_data = combined_mobility(T_range, mu0_true, mu1_true)
    mu_data_noise = mu_data * (1 + 0.05 * np.random.randn(len(T_range)))  # 5%ノイズ
    
    # フィッティング
    model = Model(combined_mobility)
    params = model.make_params(mu0=5000, mu1=3000, T0=300)
    params['T0'].vary = False  # T0は固定
    
    result = model.fit(mu_data_noise, params, T=T_range)
    
    print("温度依存性Hall測定のフィッティング結果:")
    print(result.fit_report())
    
    # 各散乱機構の寄与を計算
    mu_phonon_fit = acoustic_phonon_mobility(T_range, result.params['mu0'].value)
    mu_impurity_fit = ionized_impurity_mobility(T_range, result.params['mu1'].value)
    
    # プロット
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # 左図：移動度 vs 温度
    ax1.scatter(T_range, mu_data_noise, s=80, alpha=0.7, edgecolors='black', linewidths=1.5, label='Measured data', color='#f093fb')
    ax1.plot(T_range, result.best_fit, linewidth=2.5, label='Fit (Matthiessen)', color='#f5576c')
    ax1.plot(T_range, mu_phonon_fit, linewidth=2, linestyle='--', label='Phonon scattering (T$^{-3/2}$)', color='#ffa500')
    ax1.plot(T_range, mu_impurity_fit, linewidth=2, linestyle=':', label='Impurity scattering (T$^{3/2}$)', color='#99ccff')
    ax1.set_xlabel('Temperature T [K]', fontsize=12)
    ax1.set_ylabel('Mobility μ [cm$^2$/(V·s)]', fontsize=12)
    ax1.set_title('Temperature-Dependent Hall Mobility', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(alpha=0.3)
    ax1.set_yscale('log')
    
    # 右図：散乱率（1/μ）vs 温度
    ax2.scatter(T_range, 1/mu_data_noise, s=80, alpha=0.7, edgecolors='black', linewidths=1.5, label='Data (1/μ)', color='#f093fb')
    ax2.plot(T_range, 1/mu_phonon_fit, linewidth=2.5, label='Phonon (1/μ$_{ph}$)', color='#ffa500')
    ax2.plot(T_range, 1/mu_impurity_fit, linewidth=2.5, label='Impurity (1/μ$_{imp}$)', color='#99ccff')
    ax2.plot(T_range, 1/result.best_fit, linewidth=2.5, label='Total (sum)', color='#f5576c', linestyle='--')
    ax2.set_xlabel('Temperature T [K]', fontsize=12)
    ax2.set_ylabel('Scattering Rate 1/μ [V·s/cm$^2$]', fontsize=12)
    ax2.set_title('Matthiessen Rule: Scattering Rate Analysis', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # 結果の解釈
    print(f"\n散乱機構の解析:")
    print(f"  フォノン散乱制限移動度（室温）: {result.params['mu0'].value:.1f} cm²/(V·s)")
    print(f"  不純物散乱制限移動度（室温）: {result.params['mu1'].value:.1f} cm²/(V·s)")
    print(f"\n支配的な散乱機構:")
    print(f"  低温（< 150 K）: 不純物散乱（μ ∝ T^(3/2)）")
    print(f"  高温（> 250 K）: フォノン散乱（μ ∝ T^(-3/2)）")
    

## 2.6 完全なHallデータ処理ワークフロー

#### コード例2-6: VH → n, μ の完全解析
    
    
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    
    class HallDataProcessor:
        """完全なHallデータ処理クラス"""
    
        def __init__(self, thickness):
            """
            Parameters
            ----------
            thickness : float
                試料厚さ [m]
            """
            self.t = thickness
            self.e = 1.60218e-19  # 電荷素量 [C]
            self.data = {}
    
        def load_data(self, filename=None, mock_data=True):
            """
            測定データの読み込み
    
            Parameters
            ----------
            filename : str or None
                CSVファイル名（Noneの場合はモックデータ生成）
            mock_data : bool
                モックデータを生成するか
            """
            if mock_data:
                # モックデータ生成（温度依存性）
                T = np.array([77, 100, 150, 200, 250, 300, 350, 400])  # [K]
                I = 100e-6  # 電流 [A]
                B = 0.5  # 磁場 [T]
    
                # van der Pauw抵抗（温度依存）
                R_AB_CD = 500 + 2.0 * T
                R_BC_DA = 480 + 1.8 * T
    
                # Hall電圧（温度依存、キャリア密度変化を含む）
                V_pos = -3e-3 * (1 + 0.002 * (T - 300))
                V_neg = +2.9e-3 * (1 + 0.002 * (T - 300))
    
                self.data = pd.DataFrame({
                    'T': T,
                    'I': I,
                    'B': B,
                    'R_AB_CD': R_AB_CD,
                    'R_BC_DA': R_BC_DA,
                    'V_pos_B': V_pos,
                    'V_neg_B': V_neg
                })
            else:
                # 実データをCSVから読み込み
                self.data = pd.read_csv(filename)
    
            return self.data
    
        def calculate_sheet_resistance(self):
            """シート抵抗を計算"""
            from scipy.optimize import fsolve
    
            def vdp_eq(Rs, R1, R2):
                return np.exp(-np.pi * R1 / Rs) + np.exp(-np.pi * R2 / Rs) - 1
    
            R_s_list = []
            for _, row in self.data.iterrows():
                R1, R2 = row['R_AB_CD'], row['R_BC_DA']
                R_initial = (R1 + R2) / 2 * np.pi / np.log(2)
                R_s = fsolve(vdp_eq, R_initial, args=(R1, R2))[0]
                R_s_list.append(R_s)
    
            self.data['R_s'] = R_s_list
            self.data['sigma'] = 1 / (np.array(R_s_list) * self.t)
            self.data['rho'] = 1 / self.data['sigma']
    
        def calculate_hall_properties(self):
            """Hall特性を計算"""
            # Hall電圧（オフセット除去）
            self.data['V_H'] = 0.5 * (self.data['V_pos_B'] - self.data['V_neg_B'])
    
            # Hall係数
            self.data['R_H'] = (self.data['V_H'] * self.t) / (self.data['I'] * self.data['B'])
    
            # キャリア密度
            self.data['n'] = 1 / (np.abs(self.data['R_H']) * self.e)
    
            # 移動度
            self.data['mu'] = self.data['sigma'] * np.abs(self.data['R_H'])
    
            # キャリア種別
            self.data['carrier_type'] = ['electron' if rh < 0 else 'hole' for rh in self.data['R_H']]
    
        def plot_results(self):
            """結果の可視化"""
            fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
            T = self.data['T']
    
            # シート抵抗
            axes[0, 0].plot(T, self.data['R_s'], 'o-', linewidth=2.5, markersize=8, color='#f093fb')
            axes[0, 0].set_xlabel('Temperature [K]', fontsize=11)
            axes[0, 0].set_ylabel('Sheet Resistance [Ω/sq]', fontsize=11)
            axes[0, 0].set_title('Sheet Resistance', fontsize=12, fontweight='bold')
            axes[0, 0].grid(alpha=0.3)
    
            # 電気伝導率
            axes[0, 1].semilogy(T, self.data['sigma'], 'o-', linewidth=2.5, markersize=8, color='#f5576c')
            axes[0, 1].set_xlabel('Temperature [K]', fontsize=11)
            axes[0, 1].set_ylabel('Conductivity [S/m]', fontsize=11)
            axes[0, 1].set_title('Electrical Conductivity', fontsize=12, fontweight='bold')
            axes[0, 1].grid(alpha=0.3)
    
            # Hall電圧
            axes[0, 2].plot(T, self.data['V_H'] * 1e3, 'o-', linewidth=2.5, markersize=8, color='#ffa500')
            axes[0, 2].axhline(0, color='black', linestyle='--', linewidth=1.5)
            axes[0, 2].set_xlabel('Temperature [K]', fontsize=11)
            axes[0, 2].set_ylabel('Hall Voltage [mV]', fontsize=11)
            axes[0, 2].set_title('Hall Voltage', fontsize=12, fontweight='bold')
            axes[0, 2].grid(alpha=0.3)
    
            # Hall係数
            axes[1, 0].plot(T, self.data['R_H'], 'o-', linewidth=2.5, markersize=8, color='#99ccff')
            axes[1, 0].axhline(0, color='black', linestyle='--', linewidth=1.5)
            axes[1, 0].set_xlabel('Temperature [K]', fontsize=11)
            axes[1, 0].set_ylabel('Hall Coefficient [m³/C]', fontsize=11)
            axes[1, 0].set_title('Hall Coefficient', fontsize=12, fontweight='bold')
            axes[1, 0].grid(alpha=0.3)
    
            # キャリア密度
            axes[1, 1].semilogy(T, self.data['n'] / 1e6, 'o-', linewidth=2.5, markersize=8, color='#99ff99')
            axes[1, 1].set_xlabel('Temperature [K]', fontsize=11)
            axes[1, 1].set_ylabel('Carrier Density [cm$^{-3}$]', fontsize=11)
            axes[1, 1].set_title('Carrier Density', fontsize=12, fontweight='bold')
            axes[1, 1].grid(alpha=0.3)
    
            # 移動度
            axes[1, 2].semilogy(T, self.data['mu'] * 1e4, 'o-', linewidth=2.5, markersize=8, color='#ff9999')
            axes[1, 2].set_xlabel('Temperature [K]', fontsize=11)
            axes[1, 2].set_ylabel('Mobility [cm$^2$/(V·s)]', fontsize=11)
            axes[1, 2].set_title('Hall Mobility', fontsize=12, fontweight='bold')
            axes[1, 2].grid(alpha=0.3)
    
            plt.tight_layout()
            plt.show()
    
        def save_results(self, filename='hall_results.csv'):
            """結果をCSVに保存"""
            self.data.to_csv(filename, index=False)
            print(f"Results saved to {filename}")
    
    # 使用例
    processor = HallDataProcessor(thickness=200e-9)  # 200 nm
    processor.load_data(mock_data=True)
    processor.calculate_sheet_resistance()
    processor.calculate_hall_properties()
    
    print("Hall測定データ処理結果:")
    print(processor.data[['T', 'R_s', 'sigma', 'n', 'mu']].to_string(index=False))
    
    processor.plot_results()
    processor.save_results('hall_analysis_output.csv')
    

#### コード例2-7: Hall測定の不確かさ評価
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    def hall_measurement_uncertainty(V_H, delta_V_H, I, delta_I, B, delta_B, t, delta_t):
        """
        Hall測定の不確かさ伝播
    
        Parameters
        ----------
        V_H, delta_V_H : float
            Hall電圧とその不確かさ [V]
        I, delta_I : float
            電流とその不確かさ [A]
        B, delta_B : float
            磁場とその不確かさ [T]
        t, delta_t : float
            厚さとその不確かさ [m]
    
        Returns
        -------
        R_H, delta_R_H : float
            Hall係数とその不確かさ
        n, delta_n : float
            キャリア密度とその不確かさ
        """
        e = 1.60218e-19
    
        # Hall係数: R_H = V_H * t / (I * B)
        R_H = V_H * t / (I * B)
    
        # 不確かさ伝播（偏微分）
        # δR_H/R_H = sqrt((δV_H/V_H)^2 + (δt/t)^2 + (δI/I)^2 + (δB/B)^2)
        rel_unc_V_H = delta_V_H / np.abs(V_H)
        rel_unc_t = delta_t / t
        rel_unc_I = delta_I / I
        rel_unc_B = delta_B / B
    
        rel_unc_R_H = np.sqrt(rel_unc_V_H**2 + rel_unc_t**2 + rel_unc_I**2 + rel_unc_B**2)
        delta_R_H = np.abs(R_H) * rel_unc_R_H
    
        # キャリア密度: n = 1 / (e * R_H)
        n = 1 / (e * np.abs(R_H))
    
        # δn/n = δR_H/R_H
        delta_n = n * rel_unc_R_H
    
        return R_H, delta_R_H, n, delta_n, rel_unc_R_H
    
    # 測定例
    V_H = -5.0e-3  # [V]
    delta_V_H = 0.1e-3  # 電圧計精度 [V]
    I = 100e-6  # [A]
    delta_I = 0.5e-6  # 電流源精度 [A]
    B = 0.5  # [T]
    delta_B = 0.01  # 磁場精度 [T]
    t = 200e-9  # [m]
    delta_t = 5e-9  # 厚さ測定精度 [m]
    
    R_H, delta_R_H, n, delta_n, rel_unc = hall_measurement_uncertainty(
        V_H, delta_V_H, I, delta_I, B, delta_B, t, delta_t
    )
    
    print("Hall測定の不確かさ評価:")
    print("=" * 60)
    print("測定値:")
    print(f"  Hall電圧: ({V_H * 1e3:.2f} ± {delta_V_H * 1e3:.2f}) mV")
    print(f"  電流: ({I * 1e6:.1f} ± {delta_I * 1e6:.2f}) μA")
    print(f"  磁場: ({B:.2f} ± {delta_B:.3f}) T")
    print(f"  厚さ: ({t * 1e9:.1f} ± {delta_t * 1e9:.1f}) nm")
    print("\n結果:")
    print(f"  Hall係数: ({R_H:.3e} ± {delta_R_H:.3e}) m³/C")
    print(f"  相対不確かさ: {rel_unc * 100:.2f}%")
    print(f"  キャリア密度: ({n:.3e} ± {delta_n:.3e}) m⁻³")
    print(f"                = ({n / 1e6:.3e} ± {delta_n / 1e6:.3e}) cm⁻³")
    
    # 不確かさの寄与を可視化
    contributions = {
        'V_H': (delta_V_H / np.abs(V_H))**2,
        't': (delta_t / t)**2,
        'I': (delta_I / I)**2,
        'B': (delta_B / B)**2
    }
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    labels = list(contributions.keys())
    values = [np.sqrt(v) * 100 for v in contributions.values()]
    
    bars = ax.bar(labels, values, color=['#f093fb', '#f5576c', '#ffa500', '#99ccff'], edgecolor='black', linewidth=1.5)
    ax.set_ylabel('Relative Uncertainty Contribution [%]', fontsize=12)
    ax.set_title('Uncertainty Budget for Hall Measurement', fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3, axis='y')
    
    # バーの上に値を表示
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # 総不確かさを追加
    ax.axhline(rel_unc * 100, color='red', linestyle='--', linewidth=2, label=f'Total: {rel_unc * 100:.2f}%')
    ax.legend(fontsize=11)
    
    plt.tight_layout()
    plt.show()
    

## 2.7 演習問題

### 演習2-1: Hall電圧の計算（Easy）

Easy **問題** ：キャリア密度 $n = 1 \times 10^{22}$ m$^{-3}$、厚さ $t = 100$ nm、電流 $I = 1$ mA、磁場 $B = 0.5$ T のとき、Hall電圧 $V_H$ を計算せよ。

**解答例を表示**
    
    
    e = 1.60218e-19  # [C]
    n = 1e22  # [m^-3]
    t = 100e-9  # [m]
    I = 1e-3  # [A]
    B = 0.5  # [T]
    
    V_H = I * B / (n * e * t)
    print(f"Hall電圧 V_H = {V_H:.3e} V = {V_H * 1e3:.2f} mV")
    

**解答** ：V$_H$ = 3.12 × 10$^{-3}$ V = 3.12 mV

### 演習2-2: キャリア密度の計算（Easy）

Easy **問題** ：Hall係数 $R_H = -1.5 \times 10^{-3}$ m$^3$/C が測定された。キャリア種別とキャリア密度を求めよ。

**解答例を表示**
    
    
    import numpy as np
    
    e = 1.60218e-19  # [C]
    R_H = -1.5e-3  # [m^3/C]
    
    carrier_type = 'electron' if R_H < 0 else 'hole'
    n = 1 / (np.abs(R_H) * e)
    
    print(f"キャリア種別: {carrier_type}")
    print(f"キャリア密度 n = {n:.3e} m⁻³ = {n / 1e6:.3e} cm⁻³")
    

**解答** ：電子（n型）、n = 4.16 × 10$^{21}$ m$^{-3}$ = 4.16 × 10$^{15}$ cm$^{-3}$

### 演習2-3: 移動度の計算（Easy）

Easy **問題** ：電気伝導率 $\sigma = 1 \times 10^4$ S/m、Hall係数 $R_H = -2 \times 10^{-3}$ m$^3$/C のとき、移動度 $\mu$ を計算せよ。

**解答例を表示**
    
    
    import numpy as np
    
    sigma = 1e4  # [S/m]
    R_H = -2e-3  # [m^3/C]
    
    mu = sigma * np.abs(R_H)
    print(f"移動度 μ = {mu:.2f} m²/(V·s) = {mu * 1e4:.1f} cm²/(V·s)")
    

**解答** ：μ = 20 m$^2$/(V·s) = 200,000 cm$^2$/(V·s)（非現実的に高い→測定値の見直しが必要）

### 演習2-4: van der Pauw Hall配置の解析（Medium）

Medium **問題** ：van der Pauw測定で、$R_{\text{AB,CD}} = 950$ Ω、$R_{\text{BC,DA}} = 1050$ Ω、$V_{24}^{+B} = -4.5$ mV、$V_{24}^{-B} = +4.3$ mV、$I = 100$ μA、$B = 0.5$ T、$t = 300$ nm が得られた。$\sigma$、$R_H$、$n$、$\mu$ を計算せよ。

**解答例を表示**
    
    
    import numpy as np
    from scipy.optimize import fsolve
    
    def van_der_pauw_Rs(R1, R2):
        def eq(Rs):
            return np.exp(-np.pi * R1 / Rs) + np.exp(-np.pi * R2 / Rs) - 1
        R_init = (R1 + R2) / 2 * np.pi / np.log(2)
        return fsolve(eq, R_init)[0]
    
    # パラメータ
    R1, R2 = 950, 1050  # [Ω]
    V_pos, V_neg = -4.5e-3, 4.3e-3  # [V]
    I = 100e-6  # [A]
    B = 0.5  # [T]
    t = 300e-9  # [m]
    e = 1.60218e-19  # [C]
    
    # シート抵抗
    R_s = van_der_pauw_Rs(R1, R2)
    sigma = 1 / (R_s * t)
    
    # Hall解析
    V_H = 0.5 * (V_pos - V_neg)
    R_H = V_H * t / (I * B)
    n = 1 / (np.abs(R_H) * e)
    mu = sigma * np.abs(R_H)
    
    print(f"シート抵抗 R_s = {R_s:.2f} Ω/sq")
    print(f"電気伝導率 σ = {sigma:.2e} S/m")
    print(f"Hall係数 R_H = {R_H:.3e} m³/C")
    print(f"キャリア密度 n = {n:.2e} m⁻³ = {n / 1e6:.2e} cm⁻³")
    print(f"移動度 μ = {mu:.2e} m²/(V·s) = {mu * 1e4:.1f} cm²/(V·s)")
    

**解答** ：R$_s$ ≈ 1370 Ω/sq、σ ≈ 2.43 × 10$^3$ S/m、R$_H$ ≈ -2.64 × 10$^{-2}$ m$^3$/C、n ≈ 2.36 × 10$^{20}$ m$^{-3}$、μ ≈ 0.064 m$^2$/(V·s) = 640 cm$^2$/(V·s)

### 演習2-5: Two-Band Modelの解析（Medium）

Medium **問題** ：$n_e = 1 \times 10^{22}$ m$^{-3}$、$\mu_e = 0.5$ m$^2$/(V·s)、$n_h = 5 \times 10^{21}$ m$^{-3}$、$\mu_h = 0.05$ m$^2$/(V·s) のとき、電気伝導率 $\sigma$ とHall係数 $R_H$ を計算せよ。

**解答例を表示**
    
    
    e = 1.60218e-19  # [C]
    n_e = 1e22  # [m^-3]
    mu_e = 0.5  # [m^2/(V·s)]
    n_h = 5e21  # [m^-3]
    mu_h = 0.05  # [m^2/(V·s)]
    
    # 電気伝導率
    sigma = n_e * e * mu_e + n_h * e * mu_h
    print(f"電気伝導率 σ = {sigma:.2e} S/m")
    
    # Hall係数
    numerator = n_h * mu_h**2 - n_e * mu_e**2
    denominator = (n_h * mu_h + n_e * mu_e)**2
    R_H = numerator / (e * denominator)
    print(f"Hall係数 R_H = {R_H:.3e} m³/C")
    
    # 見かけのキャリア密度
    n_apparent = 1 / (abs(R_H) * e)
    print(f"見かけのキャリア密度: {n_apparent:.2e} m⁻³ = {n_apparent / 1e6:.2e} cm⁻³")
    print(f"  （真の電子密度 {n_e / 1e6:.2e} cm⁻³とは異なる）")
    

**解答** ：σ ≈ 8.41 × 10$^2$ S/m、R$_H$ ≈ -1.37 × 10$^{-3}$ m$^3$/C、見かけのn ≈ 4.56 × 10$^{21}$ m$^{-3}$（真の値とは異なる）

### 演習2-6: 温度依存性フィッティング（Medium）

Medium **問題** ：移動度が T = 100 K で 5000 cm$^2$/(V·s)、300 K で 1500 cm$^2$/(V·s) であった。$\mu \propto T^{-\alpha}$ モデルで指数 $\alpha$ を求めよ。

**解答例を表示**
    
    
    import numpy as np
    
    T1, mu1 = 100, 5000  # [K], [cm^2/(V·s)]
    T2, mu2 = 300, 1500  # [K], [cm^2/(V·s)]
    
    # μ ∝ T^(-α) より、log(μ) = -α log(T) + const
    # log(mu1/mu2) = -α log(T1/T2)
    # α = -log(mu1/mu2) / log(T1/T2)
    
    alpha = -np.log(mu1 / mu2) / np.log(T1 / T2)
    print(f"指数 α = {alpha:.2f}")
    print(f"モデル: μ ∝ T^(-{alpha:.2f})")
    print(f"\n解釈: α ≈ 1.5 → 音響フォノン散乱が支配的")
    

**解答** ：α ≈ 1.10（理論値 3/2 に近いが、複数の散乱機構が寄与している可能性）

### 演習2-7: 磁場依存性の解析（Hard）

Hard **問題** ：Hall電圧が磁場 B = 0, 0.2, 0.4, 0.6, 0.8, 1.0 T で V$_H$ = 0, 1.0, 2.1, 3.0, 4.1, 5.0 mV と測定された（電流 I = 100 μA、厚さ t = 200 nm）。線形フィッティングで Hall係数を求め、非線形性を評価せよ。

**解答例を表示**
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import linregress
    
    B = np.array([0, 0.2, 0.4, 0.6, 0.8, 1.0])  # [T]
    V_H = np.array([0, 1.0, 2.1, 3.0, 4.1, 5.0]) * 1e-3  # [V]
    I = 100e-6  # [A]
    t = 200e-9  # [m]
    
    # 線形フィッティング
    slope, intercept, r_value, _, std_err = linregress(B, V_H)
    
    # Hall係数
    R_H = slope * t / I
    print(f"線形フィッティング: V_H = {slope * 1e3:.3f} mV/T × B + {intercept * 1e3:.3f} mV")
    print(f"Hall係数 R_H = {R_H:.3e} m³/C")
    print(f"決定係数 R² = {r_value**2:.4f}")
    
    # 非線形性評価
    V_H_fit = slope * B + intercept
    residuals = V_H - V_H_fit
    rel_residuals = residuals / V_H_fit[1:] * 100  # 最初の点（B=0）を除く
    
    print(f"\n非線形性評価:")
    print(f"  最大残差: {np.max(np.abs(residuals[1:])) * 1e6:.2f} μV")
    print(f"  平均相対残差: {np.mean(np.abs(rel_residuals)):.2f}%")
    
    # プロット
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.scatter(B, V_H * 1e3, s=100, edgecolors='black', linewidths=2, label='Measured', color='#f093fb', zorder=5)
    ax1.plot(B, V_H_fit * 1e3, linewidth=2.5, label=f'Fit: {slope * 1e3:.3f} mV/T', color='#f5576c', linestyle='--')
    ax1.set_xlabel('Magnetic Field B [T]', fontsize=12)
    ax1.set_ylabel('Hall Voltage V$_H$ [mV]', fontsize=12)
    ax1.set_title('Hall Voltage vs Magnetic Field', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(alpha=0.3)
    
    ax2.scatter(B[1:], residuals[1:] * 1e6, s=100, edgecolors='black', linewidths=2, color='#ffa500')
    ax2.axhline(0, color='black', linestyle='--', linewidth=1.5)
    ax2.set_xlabel('Magnetic Field B [T]', fontsize=12)
    ax2.set_ylabel('Residuals [μV]', fontsize=12)
    ax2.set_title('Fit Residuals', fontsize=14, fontweight='bold')
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    

**解答** ：R$_H$ ≈ 1.00 × 10$^{-2}$ m$^3$/C、R$^2$ ≈ 0.9996（良好な線形性）、最大残差 < 50 μV（測定精度内）

### 演習2-8: 不確かさ伝播（Hard）

Hard **問題** ：Hall電圧 $V_H = (5.0 \pm 0.2)$ mV、電流 $I = (100 \pm 1)$ μA、磁場 $B = (0.50 \pm 0.02)$ T、厚さ $t = (200 \pm 10)$ nm のとき、キャリア密度 $n$ の不確かさ $\Delta n$ を計算せよ。どのパラメータが最も影響するか評価せよ。

**解答例を表示**
    
    
    import numpy as np
    
    V_H, dV_H = 5.0e-3, 0.2e-3  # [V]
    I, dI = 100e-6, 1e-6  # [A]
    B, dB = 0.50, 0.02  # [T]
    t, dt = 200e-9, 10e-9  # [m]
    e = 1.60218e-19  # [C]
    
    # n = 1 / (e * R_H) = I * B / (e * V_H * t)
    n = I * B / (e * V_H * t)
    
    # 相対不確かさの寄与
    rel_V_H = (dV_H / V_H)**2
    rel_I = (dI / I)**2
    rel_B = (dB / B)**2
    rel_t = (dt / t)**2
    
    rel_unc_total = np.sqrt(rel_V_H + rel_I + rel_B + rel_t)
    dn = n * rel_unc_total
    
    print(f"キャリア密度: n = {n:.3e} m⁻³ = {n / 1e6:.3e} cm⁻³")
    print(f"不確かさ: Δn = {dn:.3e} m⁻³ = {dn / 1e6:.3e} cm⁻³")
    print(f"相対不確かさ: {rel_unc_total * 100:.2f}%")
    print(f"\n不確かさの寄与:")
    print(f"  V_H: {np.sqrt(rel_V_H) * 100:.2f}%")
    print(f"  I:   {np.sqrt(rel_I) * 100:.2f}%")
    print(f"  B:   {np.sqrt(rel_B) * 100:.2f}%")
    print(f"  t:   {np.sqrt(rel_t) * 100:.2f}%")
    print(f"\n結論: 厚さ t の不確かさが最も大きく影響（{np.sqrt(rel_t) * 100:.1f}%）")
    

**解答** ：n = (6.24 ± 0.42) × 10$^{21}$ m$^{-3}$、相対不確かさ 6.7%、厚さ測定が最も重要（5.0%寄与）

### 演習2-9: 実験計画（Hard）

Hard **問題** ：未知の薄膜半導体（厚さ 100 nm）のキャリア種別、密度、移動度を決定する実験計画を立案せよ。必要な測定、温度範囲、磁場範囲、データ解析手法を具体的に説明せよ。

**解答例を表示**

**実験計画** ：

  1. **試料準備** ： 
     * van der Pauw配置：四隅に8接点（電流用4つ、電圧用4つ）
     * 接点サイズ < 0.5 mm、オーミック接触確認（I-V特性が線形）
  2. **室温測定** （T = 300 K）： 
     * **シート抵抗測定** （B = 0）：$R_{\text{AB,CD}}$、$R_{\text{BC,DA}}$ → $R_s$、$\sigma$ 計算
     * **Hall測定** ：B = ±0.5 T で $V_H$ 測定（オフセット除去）
     * → キャリア種別（$R_H$ の符号）、キャリア密度 $n$、移動度 $\mu$ を決定
  3. **磁場依存性測定** （室温）： 
     * B = 0 → 1 T まで 0.1 T ステップで Hall電圧測定
     * 線形性確認（非線形 → 多キャリア系の可能性）
  4. **温度依存性測定** ： 
     * 温度範囲：77 K（液体窒素）〜 400 K
     * 測定点：20-25 K 間隔、各温度で熱平衡待機（10-15分）
     * 各温度で：シート抵抗 + Hall測定（B = ±0.5 T）
  5. **データ解析** ： 
     * $n(T)$、$\mu(T)$ のプロット作成
     * 半導体か金属か判定（$\rho$ の温度依存性）
     * 半導体の場合：Arrheniusプロット（$\ln n$ vs $1/T$）→ 活性化エネルギー
     * 移動度の温度依存性フィッティング（Matthiessenの法則、$\mu \propto T^{-\alpha}$）
     * 散乱機構の特定（音響フォノン、不純物、粒界など）

**期待される結果** ：

  * **n型半導体** ：$R_H < 0$、$n \sim 10^{15}$-10$^{18}$ cm$^{-3}$、$\mu \sim 100$-5000 cm$^2$/(V·s)、温度上昇で $n$ 増加（熱励起）
  * **p型半導体** ：$R_H > 0$、同様の密度・移動度範囲
  * **金属的材料** ：$n \sim 10^{21}$-10$^{23}$ cm$^{-3}$、$\mu$ は温度上昇で減少（フォノン散乱）

## 2.8 学習の確認

以下のチェックリストで理解度を確認しましょう：

### 基本理解

  * Lorentz力からHall電圧の式を導出できる
  * Hall係数 $R_H = 1/(ne)$ とキャリア密度の関係を理解している
  * 移動度 $\mu = \sigma R_H$ の物理的意味を説明できる
  * van der Pauw Hall配置の測定手順を理解している
  * 磁場反転測定の目的（オフセット除去）を説明できる

### 実践スキル

  * Hall電圧からキャリア密度を計算できる
  * van der Pauw Hall測定データを完全に解析できる（$\sigma$、$R_H$、$n$、$\mu$）
  * Pythonで完全なHallデータ処理ワークフローを実装できる
  * 不確かさ伝播を計算し、測定精度を評価できる
  * 温度依存性データから散乱機構を推定できる

### 応用力

  * two-band modelで多キャリア系を解析できる
  * 移動度の温度依存性から散乱機構を特定できる
  * 測定の非線形性を評価し、原因を推定できる
  * 実験計画を立案し、適切な測定条件を決定できる

## 2.9 参考文献

  1. Hall, E. H. (1879). _On a New Action of the Magnet on Electric Currents_. American Journal of Mathematics, 2(3), 287-292. - Hall効果の発見論文
  2. van der Pauw, L. J. (1958). _A method of measuring the resistivity and Hall coefficient on lamellae of arbitrary shape_. Philips Technical Review, 20(8), 220-224. - van der Pauw Hall測定法の原論文
  3. Look, D. C. (1989). _Electrical Characterization of GaAs Materials and Devices_. Wiley. - 半導体Hall測定の実践的テキスト
  4. Putley, E. H. (1960). _The Hall Effect and Related Phenomena_. Butterworths. - Hall効果の包括的解説
  5. Ashcroft, N. W., & Mermin, N. D. (1976). _Solid State Physics_ (Chapter 2: The Sommerfeld Theory of Metals). Holt, Rinehart and Winston. - Drudeモデルとキャリア輸送理論
  6. Schroder, D. K. (2006). _Semiconductor Material and Device Characterization_ (3rd ed., Chapter 2: Resistivity). Wiley-Interscience. - Hall測定技術の詳細
  7. Popović, R. S. (2004). _Hall Effect Devices_ (2nd ed.). Institute of Physics Publishing. - Hall効果デバイスと測定技術

## 2.10 次章へ

次章では、**磁気測定** の原理と実践を学びます。VSM（Vibrating Sample Magnetometer）やSQUID（超伝導量子干渉計）を用いた磁化測定、M-H曲線解析、磁気異方性評価、PPMS（Physical Property Measurement System）による統合測定技術を習得します。
