---
title: "第1章: 電気伝導測定の基礎"
chapter_title: "第1章: 電気伝導測定の基礎"
subtitle: Drudeモデルから四端子測定法、van der Pauw法、温度依存性解析まで
reading_time: 40-50分
difficulty: 中級
code_examples: 7
---

電気伝導測定は、材料の電気的特性を定量的に評価する基本的な手法です。この章では、Drudeモデルによる電気伝導の理論的基礎、二端子測定と四端子測定の違い、van der Pauw法による任意形状試料の測定、温度依存性から得られるキャリア散乱機構の情報を学び、Pythonで実践的なデータ解析を行います。 

## 学習目標

この章を読むことで、以下を習得できます：

  * ✅ Drudeモデルを理解し、電気伝導率の式 $\sigma = ne^2\tau/m$ を導出できる
  * ✅ 二端子測定と四端子測定の原理と違いを説明できる
  * ✅ van der Pauw法の理論を理解し、Pythonで実装できる
  * ✅ 接触抵抗の影響を評価し、適切な補正を行える
  * ✅ 温度依存性データからキャリア散乱機構を解析できる
  * ✅ シート抵抗とバルク抵抗率の関係を理解できる
  * ✅ Pythonでフィッティングと不確かさ評価ができる

## 1.1 電気伝導のDrudeモデル

### 1.1.1 Drudeモデルの基礎

**Drudeモデル** （1900年、Paul Drude提唱）は、金属や高ドープ半導体の電気伝導を説明する古典的な理論です。このモデルでは、電子を自由に動き回る「自由電子ガス」として扱います。

**基本仮定** ：

  * 電子は原子核や他の電子と**散乱** するまで自由に運動する
  * 散乱は平均緩和時間 $\tau$ で起こる（散乱確率 $\propto 1/\tau$）
  * 散乱後、電子の速度はランダムな方向を向く（電場による drift はリセット）

**電気伝導率の導出** ：

電場 $\vec{E}$ がかかると、電子は加速される：

$$ m\frac{d\vec{v}}{dt} = -e\vec{E} - \frac{m\vec{v}}{\tau} $$ 

ここで、$m$ は電子質量、$e$ は電荷素量（$e > 0$）、$\tau$ は散乱緩和時間です。定常状態（$d\vec{v}/dt = 0$）で：

$$ \vec{v}_{\text{drift}} = -\frac{e\tau}{m}\vec{E} $$ 

電流密度 $\vec{j}$ は、キャリア密度 $n$、電荷 $-e$、ドリフト速度 $\vec{v}_{\text{drift}}$ の積：

$$ \vec{j} = -ne\vec{v}_{\text{drift}} = \frac{ne^2\tau}{m}\vec{E} $$ 

**電気伝導率** $\sigma$ と**抵抗率** $\rho$ は：

$$ \sigma = \frac{ne^2\tau}{m}, \quad \rho = \frac{1}{\sigma} = \frac{m}{ne^2\tau} $$ 

**移動度** $\mu$ は、単位電場あたりのドリフト速度：

$$ \mu = \frac{|v_{\text{drift}}|}{E} = \frac{e\tau}{m} $$ 

したがって、電気伝導率は次のようにも表現できます：

$$ \sigma = ne\mu $$ 

#### コード例1-1: Drudeモデルによる電気伝導率計算
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    def drude_conductivity(n, tau, m=9.10938e-31):
        """
        Drudeモデルによる電気伝導率を計算
    
        Parameters
        ----------
        n : float or array-like
            キャリア密度 [m^-3]
        tau : float or array-like
            散乱緩和時間 [s]
        m : float
            有効質量 [kg]（デフォルト：自由電子質量）
    
        Returns
        -------
        sigma : float or array-like
            電気伝導率 [S/m]
        """
        e = 1.60218e-19  # 電荷素量 [C]
        sigma = n * e**2 * tau / m
        return sigma
    
    # 典型的な金属（銅）のパラメータ
    n_Cu = 8.5e28  # キャリア密度 [m^-3]（銅の自由電子密度）
    tau_Cu = 2.5e-14  # 散乱緩和時間 [s]（室温）
    
    sigma_Cu = drude_conductivity(n_Cu, tau_Cu)
    rho_Cu = 1 / sigma_Cu
    
    print(f"銅の電気伝導率: {sigma_Cu:.3e} S/m")
    print(f"銅の抵抗率: {rho_Cu:.3e} Ω·m = {rho_Cu * 1e8:.2f} μΩ·cm")
    print(f"実験値（室温）: ρ ≈ 1.68 μΩ·cm")
    
    # キャリア密度依存性
    n_range = np.logspace(26, 30, 100)  # [m^-3]
    tau_fixed = 1e-14  # [s]
    
    sigma_n = drude_conductivity(n_range, tau_fixed)
    
    # 散乱緩和時間依存性
    n_fixed = 1e28  # [m^-3]
    tau_range = np.logspace(-15, -12, 100)  # [s]
    
    sigma_tau = drude_conductivity(n_fixed, tau_range)
    
    # プロット
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # 左図：キャリア密度依存性
    ax1.loglog(n_range, sigma_n, linewidth=2.5, color='#f093fb')
    ax1.scatter([n_Cu], [sigma_Cu], s=150, c='#f5576c', edgecolors='black', linewidth=2, zorder=5, label='Cu (room temp)')
    ax1.set_xlabel('Carrier Density n [m$^{-3}$]', fontsize=12)
    ax1.set_ylabel('Conductivity σ [S/m]', fontsize=12)
    ax1.set_title('Conductivity vs Carrier Density\n(τ = 10 fs)', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(alpha=0.3, which='both')
    
    # 右図：散乱緩和時間依存性
    ax2.loglog(tau_range * 1e15, sigma_tau, linewidth=2.5, color='#f093fb')
    ax2.scatter([tau_Cu * 1e15], [sigma_Cu], s=150, c='#f5576c', edgecolors='black', linewidth=2, zorder=5, label='Cu (room temp)')
    ax2.set_xlabel('Scattering Relaxation Time τ [fs]', fontsize=12)
    ax2.set_ylabel('Conductivity σ [S/m]', fontsize=12)
    ax2.set_title('Conductivity vs Relaxation Time\n(n = 10$^{28}$ m$^{-3}$)', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(alpha=0.3, which='both')
    
    plt.tight_layout()
    plt.show()
    

**出力解釈** ：

  * 銅の電気伝導率: 約 $5.8 \times 10^7$ S/m（実験値とよく一致）
  * 電気伝導率は $n$ と $\tau$ に比例（$\sigma \propto n\tau$）
  * 金属では $n$ はほぼ一定なので、温度依存性は主に $\tau$ に起因

### 1.1.2 温度依存性と散乱機構

散乱緩和時間 $\tau$ は、散乱機構に依存して温度により変化します。

散乱機構 | 温度依存性 | 支配的な温度域 | 材料例  
---|---|---|---  
**フォノン散乱** | $\rho \propto T$（高温） | 室温以上 | 純金属  
**不純物散乱** | $\rho = \rho_0$（定数） | 低温 | 合金、ドープ半導体  
**粒界散乱** | 温度依存性弱い | 全温度域 | 多結晶材料  
**電子-電子散乱** | $\rho \propto T^2$ | 極低温 | フェルミ液体  
  
**Matthiessenの法則** ：複数の散乱機構が独立に働く場合、総抵抗率は各機構の寄与の和：

$$ \rho(T) = \rho_0 + \rho_{\text{phonon}}(T) + \rho_{\text{other}}(T) $$ 

ここで、$\rho_0$ は残留抵抗率（不純物・欠陥による、温度に依存しない）、$\rho_{\text{phonon}}(T)$ はフォノン散乱による温度依存項です。

## 1.2 二端子測定と四端子測定

### 1.2.1 二端子測定の問題点

二端子測定では、電流を流す端子と電圧を測定する端子が同じであるため、**接触抵抗** や**配線抵抗** が測定値に含まれてしまいます。
    
    
    ```mermaid
    flowchart LR
        A[電流源] -->|I| B[接点1R_c1]
        B --> C[試料R_sample]
        C --> D[接点2R_c2]
        D -->|I| E[電圧計]
        E --> A
    
        style A fill:#99ccff,stroke:#0066cc,stroke-width:2px
        style E fill:#99ccff,stroke:#0066cc,stroke-width:2px
        style C fill:#f093fb,stroke:#f5576c,stroke-width:2px,color:#fff
        style B fill:#ffeb99,stroke:#ffa500,stroke-width:2px
        style D fill:#ffeb99,stroke:#ffa500,stroke-width:2px
    ```

測定される電圧：

$$ V_{\text{measured}} = I(R_{\text{c1}} + R_{\text{sample}} + R_{\text{c2}}) $$ 

接触抵抗 $R_c$ は、試料抵抗 $R_{\text{sample}}$ よりも大きい場合があり（特に薄膜や半導体）、正確な測定の妨げとなります。

### 1.2.2 四端子測定法（Kelvin測定）

**四端子測定法** では、電流端子と電圧端子を分離することで、接触抵抗の影響を排除します。
    
    
    ```mermaid
    flowchart LR
        A[電流源] -->|I| B[電流接点1]
        B --> C[試料R_sample]
        C --> D[電流接点2]
        D -->|I| A
    
        E[高入力インピーダンス電圧計] -.->|V+| F[電圧接点3]
        F --> C
        C --> G[電圧接点4]
        G -.->|V-| E
    
        style A fill:#99ccff,stroke:#0066cc,stroke-width:2px
        style E fill:#99ccff,stroke:#0066cc,stroke-width:2px
        style C fill:#f093fb,stroke:#f5576c,stroke-width:2px,color:#fff
    ```

電圧計は高入力インピーダンス（理想的には無限大）なので、電圧端子にはほとんど電流が流れません。したがって、電圧端子の接触抵抗による電圧降下はゼロで、試料内部の電圧のみを測定できます：

$$ V_{\text{sample}} = I \cdot R_{\text{sample}} $$ 

**四端子測定の利点** ：

  * 接触抵抗の影響を受けない
  * 配線抵抗の影響も排除できる
  * 低抵抗材料の高精度測定が可能（μΩオーダーも測定可能）

#### コード例1-2: 二端子 vs 四端子測定のシミュレーション
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    def two_terminal_measurement(R_sample, R_contact, I):
        """
        二端子測定をシミュレート
        """
        V_measured = I * (2 * R_contact + R_sample)
        R_measured = V_measured / I
        return R_measured
    
    def four_terminal_measurement(R_sample, I):
        """
        四端子測定をシミュレート
        """
        V_sample = I * R_sample
        R_measured = V_sample / I
        return R_measured
    
    # パラメータ
    R_sample = 1.0  # 試料抵抗 [Ω]
    R_contact_range = np.linspace(0, 5, 100)  # 接触抵抗 [Ω]
    I = 0.1  # 電流 [A]
    
    # 測定値計算
    R_2terminal = [two_terminal_measurement(R_sample, Rc, I) for Rc in R_contact_range]
    R_4terminal = [four_terminal_measurement(R_sample, I) for Rc in R_contact_range]
    
    # プロット
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(R_contact_range, R_2terminal, linewidth=2.5, label='2-terminal (with contact resistance)', color='#ffa500')
    ax.plot(R_contact_range, R_4terminal, linewidth=2.5, label='4-terminal (no contact resistance)', color='#f093fb', linestyle='--')
    ax.axhline(y=R_sample, color='black', linestyle=':', linewidth=1.5, label=f'True sample resistance = {R_sample} Ω')
    
    ax.set_xlabel('Contact Resistance R$_c$ [Ω]', fontsize=12)
    ax.set_ylabel('Measured Resistance [Ω]', fontsize=12)
    ax.set_title('2-Terminal vs 4-Terminal Measurement', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='upper left')
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # 具体例
    R_contact_example = 2.0  # [Ω]
    R_2t = two_terminal_measurement(R_sample, R_contact_example, I)
    R_4t = four_terminal_measurement(R_sample, I)
    
    print(f"試料抵抗: {R_sample} Ω")
    print(f"接触抵抗: {R_contact_example} Ω（各接点）")
    print(f"二端子測定値: {R_2t:.2f} Ω（誤差: {(R_2t - R_sample) / R_sample * 100:.1f}%）")
    print(f"四端子測定値: {R_4t:.2f} Ω（誤差: 0%）")
    

## 1.3 van der Pauw法

### 1.3.1 van der Pauw定理

**van der Pauw法** （1958年、L.J. van der Pauw提唱）は、任意形状の薄膜試料のシート抵抗を測定できる強力な手法です。試料の形状を特定の形（矩形、円形など）にする必要がなく、4つの接点さえあれば測定可能です。

**条件** ：

  * 試料は平面的で、厚さ $t$ が一様
  * 試料に穴や欠陥がない（単連結）
  * 4つの接点が試料の端部に配置されている
  * 接点は十分小さい

**van der Pauw定理** ：

4つの接点 A, B, C, D を試料周辺に配置し、次の2つの抵抗を測定します：

$$ R_{\text{AB,CD}} = \frac{V_{\text{CD}}}{I_{\text{AB}}} \quad \text{（ABから電流、CDで電圧測定）} $$ $$ R_{\text{BC,DA}} = \frac{V_{\text{DA}}}{I_{\text{BC}}} \quad \text{（BCから電流、DAで電圧測定）} $$ 

van der Pauwの定理により、シート抵抗 $R_s$ は次の式を満たします：

$$ \exp\left(-\frac{\pi R_{\text{AB,CD}}}{R_s}\right) + \exp\left(-\frac{\pi R_{\text{BC,DA}}}{R_s}\right) = 1 $$ 

この式を $R_s$ について解くことで、シート抵抗が求まります。

**シート抵抗とバルク抵抗率の関係** ：

$$ R_s = \frac{\rho}{t} $$ 

ここで、$\rho$ はバルク抵抗率、$t$ は試料の厚さです。$R_s$ の単位は Ω/□（オーム・パー・スクエア）で表されます。

#### コード例1-3: van der Pauw法によるシート抵抗計算
    
    
    import numpy as np
    from scipy.optimize import fsolve
    import matplotlib.pyplot as plt
    
    def van_der_pauw_equation(Rs, R1, R2):
        """
        van der Pauw方程式: exp(-π R1/Rs) + exp(-π R2/Rs) = 1
        """
        return np.exp(-np.pi * R1 / Rs) + np.exp(-np.pi * R2 / Rs) - 1
    
    def calculate_sheet_resistance(R_AB_CD, R_BC_DA):
        """
        van der Pauw法によりシート抵抗を計算
    
        Parameters
        ----------
        R_AB_CD : float
            抵抗 R_AB,CD [Ω]
        R_BC_DA : float
            抵抗 R_BC,DA [Ω]
    
        Returns
        -------
        R_s : float
            シート抵抗 [Ω/sq]
        """
        # 初期推定値：平均抵抗
        R_initial = (R_AB_CD + R_BC_DA) / 2 * np.pi / np.log(2)
    
        # 数値的に解く
        R_s = fsolve(van_der_pauw_equation, R_initial, args=(R_AB_CD, R_BC_DA))[0]
    
        return R_s
    
    # 測定例1：正方形試料（対称配置）
    R1 = 100  # [Ω]
    R2 = 100  # [Ω]
    
    R_s1 = calculate_sheet_resistance(R1, R2)
    print("例1：正方形試料（対称配置）")
    print(f"  R_AB,CD = {R1:.1f} Ω")
    print(f"  R_BC,DA = {R2:.1f} Ω")
    print(f"  シート抵抗 R_s = {R_s1:.2f} Ω/sq")
    print(f"  簡略式 R_s ≈ (π/ln2)(R1+R2)/2 = {np.pi / np.log(2) * (R1 + R2) / 2:.2f} Ω/sq")
    
    # 測定例2：非対称配置
    R1 = 120  # [Ω]
    R2 = 80   # [Ω]
    
    R_s2 = calculate_sheet_resistance(R1, R2)
    print("\n例2：非対称配置")
    print(f"  R_AB,CD = {R1:.1f} Ω")
    print(f"  R_BC,DA = {R2:.1f} Ω")
    print(f"  シート抵抗 R_s = {R_s2:.2f} Ω/sq")
    
    # 厚さからバルク抵抗率を計算
    t = 100e-9  # 厚さ 100 nm
    rho1 = R_s1 * t
    rho2 = R_s2 * t
    
    print(f"\n厚さ t = {t * 1e9:.0f} nm とすると：")
    print(f"  例1のバルク抵抗率 ρ = {rho1:.3e} Ω·m = {rho1 * 1e8:.2f} μΩ·cm")
    print(f"  例2のバルク抵抗率 ρ = {rho2:.3e} Ω·m = {rho2 * 1e8:.2f} μΩ·cm")
    
    # van der Pauw方程式の可視化
    R1_range = np.linspace(50, 150, 50)
    R2_range = np.linspace(50, 150, 50)
    R1_mesh, R2_mesh = np.meshgrid(R1_range, R2_range)
    
    R_s_mesh = np.zeros_like(R1_mesh)
    for i in range(R1_mesh.shape[0]):
        for j in range(R1_mesh.shape[1]):
            R_s_mesh[i, j] = calculate_sheet_resistance(R1_mesh[i, j], R2_mesh[i, j])
    
    # プロット
    fig, ax = plt.subplots(figsize=(10, 8))
    
    contour = ax.contourf(R1_mesh, R2_mesh, R_s_mesh, levels=20, cmap='plasma')
    cbar = plt.colorbar(contour, ax=ax)
    cbar.set_label('Sheet Resistance R$_s$ [Ω/sq]', fontsize=12)
    
    ax.scatter([R1], [R2], s=200, c='white', edgecolors='black', linewidth=2, marker='o', label='Example 2')
    ax.set_xlabel('R$_{AB,CD}$ [Ω]', fontsize=12)
    ax.set_ylabel('R$_{BC,DA}$ [Ω]', fontsize=12)
    ax.set_title('van der Pauw Method: Sheet Resistance Map', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3, color='white')
    
    plt.tight_layout()
    plt.show()
    

**出力解釈** ：

  * 対称配置（$R_1 = R_2$）の場合、簡略式 $R_s \approx \frac{\pi}{\ln 2}R_1$ がよく使われる
  * 非対称配置でも、van der Pauw方程式を数値的に解けば正確な $R_s$ が得られる
  * シート抵抗から厚さを掛けることで、バルク抵抗率が得られる

## 1.4 温度依存性測定とフィッティング

### 1.4.1 金属の温度依存性（Bloch-Grüneisenモデル）

金属の抵抗率は、低温では不純物散乱（温度に依存しない $\rho_0$）、高温ではフォノン散乱（$T$ に比例）が支配的です：

$$ \rho(T) = \rho_0 + A T $$ 

ここで、$A$ はフォノン散乱に関係する係数です。

### 1.4.2 半導体の温度依存性（Arrheniusプロット）

半導体では、キャリア密度が温度により変化します（熱励起）：

$$ n(T) \propto \exp\left(-\frac{E_a}{k_B T}\right) $$ 

抵抗率は：

$$ \rho(T) = \rho_0 \exp\left(\frac{E_a}{k_B T}\right) $$ 

$\ln \rho$ vs $1/T$ のプロット（Arrheniusプロット）で、活性化エネルギー $E_a$ が求まります。

#### コード例1-4: 温度依存性データのフィッティング
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    from lmfit import Model
    
    # 金属モデル：ρ(T) = ρ₀ + A*T
    def metal_resistivity(T, rho0, A):
        return rho0 + A * T
    
    # 半導体モデル：ρ(T) = ρ₀ * exp(Ea / (kB * T))
    def semiconductor_resistivity(T, rho0, Ea):
        kB = 8.617333e-5  # Boltzmann定数 [eV/K]
        return rho0 * np.exp(Ea / (kB * T))
    
    # シミュレーションデータ生成
    # 金属（銅）
    T_metal = np.linspace(50, 400, 30)  # [K]
    rho0_true = 0.5e-8  # [Ω·m]
    A_true = 5e-11  # [Ω·m/K]
    rho_metal = metal_resistivity(T_metal, rho0_true, A_true) * (1 + 0.02 * np.random.randn(len(T_metal)))  # 2%ノイズ
    
    # 半導体（シリコン）
    T_semi = np.linspace(300, 600, 30)  # [K]
    rho0_semi_true = 1e-5  # [Ω·m]
    Ea_true = 0.5  # [eV]
    rho_semi = semiconductor_resistivity(T_semi, rho0_semi_true, Ea_true) * (1 + 0.05 * np.random.randn(len(T_semi)))  # 5%ノイズ
    
    # フィッティング：金属
    metal_model = Model(metal_resistivity)
    metal_params = metal_model.make_params(rho0=1e-8, A=1e-11)
    metal_result = metal_model.fit(rho_metal, metal_params, T=T_metal)
    
    print("金属のフィッティング結果:")
    print(metal_result.fit_report())
    
    # フィッティング：半導体
    semi_model = Model(semiconductor_resistivity)
    semi_params = semi_model.make_params(rho0=1e-6, Ea=0.6)
    semi_result = semi_model.fit(rho_semi, semi_params, T=T_semi)
    
    print("\n半導体のフィッティング結果:")
    print(semi_result.fit_report())
    
    # プロット
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 金属：ρ vs T
    axes[0, 0].scatter(T_metal, rho_metal * 1e8, s=80, alpha=0.7, edgecolors='black', linewidths=1.5, label='Data (with noise)', color='#f093fb')
    axes[0, 0].plot(T_metal, metal_result.best_fit * 1e8, linewidth=2.5, label='Fit: ρ = ρ₀ + AT', color='#f5576c')
    axes[0, 0].set_xlabel('Temperature T [K]', fontsize=12)
    axes[0, 0].set_ylabel('Resistivity ρ [μΩ·cm]', fontsize=12)
    axes[0, 0].set_title('Metal (Cu): Resistivity vs Temperature', fontsize=13, fontweight='bold')
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(alpha=0.3)
    
    # 金属：残差
    residuals_metal = rho_metal - metal_result.best_fit
    axes[0, 1].scatter(T_metal, residuals_metal * 1e8, s=80, alpha=0.7, edgecolors='black', linewidths=1.5, color='#99ccff')
    axes[0, 1].axhline(0, color='black', linestyle='--', linewidth=1.5)
    axes[0, 1].set_xlabel('Temperature T [K]', fontsize=12)
    axes[0, 1].set_ylabel('Residuals [μΩ·cm]', fontsize=12)
    axes[0, 1].set_title('Fit Residuals (Metal)', fontsize=13, fontweight='bold')
    axes[0, 1].grid(alpha=0.3)
    
    # 半導体：ρ vs T
    axes[1, 0].scatter(T_semi, rho_semi, s=80, alpha=0.7, edgecolors='black', linewidths=1.5, label='Data (with noise)', color='#ffa500')
    axes[1, 0].plot(T_semi, semi_result.best_fit, linewidth=2.5, label='Fit: ρ = ρ₀exp(Ea/kT)', color='#ff6347')
    axes[1, 0].set_xlabel('Temperature T [K]', fontsize=12)
    axes[1, 0].set_ylabel('Resistivity ρ [Ω·m]', fontsize=12)
    axes[1, 0].set_title('Semiconductor (Si): Resistivity vs Temperature', fontsize=13, fontweight='bold')
    axes[1, 0].legend(fontsize=10)
    axes[1, 0].grid(alpha=0.3)
    axes[1, 0].set_yscale('log')
    
    # 半導体：Arrheniusプロット
    axes[1, 1].scatter(1000 / T_semi, np.log(rho_semi), s=80, alpha=0.7, edgecolors='black', linewidths=1.5, label='Data', color='#ffa500')
    axes[1, 1].plot(1000 / T_semi, np.log(semi_result.best_fit), linewidth=2.5, label='Fit (Arrhenius)', color='#ff6347')
    axes[1, 1].set_xlabel('1000/T [K$^{-1}$]', fontsize=12)
    axes[1, 1].set_ylabel('ln(ρ)', fontsize=12)
    axes[1, 1].set_title('Arrhenius Plot (Semiconductor)', fontsize=13, fontweight='bold')
    axes[1, 1].legend(fontsize=10)
    axes[1, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    

## 1.5 接触抵抗の評価と補正

### 1.5.1 Transfer Length Method (TLM)

**TLM（Transfer Length Method）** は、接触抵抗を定量的に評価する手法です。異なる間隔の接点対を作成し、測定抵抗と接点間距離の関係から接触抵抗を求めます。

測定抵抗 $R_{\text{total}}$ は：

$$ R_{\text{total}} = 2R_c + R_s \frac{L}{W} $$ 

ここで、$R_c$ は接触抵抗、$R_s$ はシート抵抗、$L$ は接点間距離、$W$ は試料幅です。$R_{\text{total}}$ を $L$ に対してプロットすると、切片から $2R_c$ が、傾きから $R_s/W$ が得られます。

#### コード例1-5: TLM法による接触抵抗評価
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import linregress
    
    # パラメータ
    R_s = 50  # シート抵抗 [Ω/sq]
    R_c = 10  # 接触抵抗 [Ω]
    W = 0.01  # 試料幅 [m] = 1 cm
    
    # 接点間距離
    L_values = np.array([0.001, 0.002, 0.003, 0.005, 0.010])  # [m]
    
    # 測定抵抗（ノイズあり）
    R_total = 2 * R_c + R_s * L_values / W
    R_total_noise = R_total * (1 + 0.03 * np.random.randn(len(L_values)))  # 3%ノイズ
    
    # 線形フィッティング
    slope, intercept, r_value, p_value, std_err = linregress(L_values * 1000, R_total_noise)
    
    R_c_fit = intercept / 2
    R_s_fit = slope * W * 1000  # 傾き × W
    
    print("TLM法による接触抵抗評価:")
    print(f"  真の接触抵抗 R_c = {R_c:.2f} Ω")
    print(f"  フィットから求めた R_c = {R_c_fit:.2f} Ω")
    print(f"  真のシート抵抗 R_s = {R_s:.2f} Ω/sq")
    print(f"  フィットから求めた R_s = {R_s_fit:.2f} Ω/sq")
    print(f"  決定係数 R² = {r_value**2:.4f}")
    
    # プロット
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.scatter(L_values * 1000, R_total_noise, s=150, edgecolors='black', linewidths=2, label='Measured data', color='#f093fb', zorder=5)
    L_fit = np.linspace(0, np.max(L_values) * 1000, 100)
    R_fit = slope * L_fit + intercept
    ax.plot(L_fit, R_fit, linewidth=2.5, label=f'Linear fit: R = {slope:.2f}L + {intercept:.2f}', color='#f5576c', linestyle='--')
    
    # 切片を強調
    ax.axhline(y=intercept, color='orange', linestyle=':', linewidth=2, label=f'Intercept = 2R$_c$ = {intercept:.2f} Ω')
    ax.scatter([0], [intercept], s=200, c='orange', edgecolors='black', linewidth=2, marker='s', zorder=5)
    
    ax.set_xlabel('Contact Spacing L [mm]', fontsize=12)
    ax.set_ylabel('Total Resistance R$_{total}$ [Ω]', fontsize=12)
    ax.set_title('Transfer Length Method (TLM)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    ax.set_xlim(left=0)
    
    plt.tight_layout()
    plt.show()
    

## 1.6 演習問題

### 演習1-1: Drudeモデルの計算（Easy）

Easy **問題** ：金のキャリア密度 $n = 5.9 \times 10^{28}$ m$^{-3}$、散乱緩和時間 $\tau = 3 \times 10^{-14}$ s として、電気伝導率と抵抗率を計算せよ。

**解答例を表示**
    
    
    import numpy as np
    
    # 定数
    e = 1.60218e-19  # [C]
    m = 9.10938e-31  # [kg]
    
    # 金のパラメータ
    n = 5.9e28  # [m^-3]
    tau = 3e-14  # [s]
    
    # 電気伝導率
    sigma = n * e**2 * tau / m
    rho = 1 / sigma
    
    print(f"電気伝導率 σ = {sigma:.3e} S/m")
    print(f"抵抗率 ρ = {rho:.3e} Ω·m = {rho * 1e8:.2f} μΩ·cm")
    print(f"実験値（室温）: ρ ≈ 2.44 μΩ·cm")
    

**解答** ：σ ≈ 4.54 × 10$^7$ S/m、ρ ≈ 2.20 × 10$^{-8}$ Ω·m = 2.20 μΩ·cm（実験値とよく一致）

### 演習1-2: 移動度の計算（Easy）

Easy **問題** ：散乱緩和時間 $\tau = 1 \times 10^{-14}$ s の電子の移動度 $\mu$ を計算せよ。

**解答例を表示**
    
    
    e = 1.60218e-19  # [C]
    m = 9.10938e-31  # [kg]
    tau = 1e-14  # [s]
    
    mu = e * tau / m
    print(f"移動度 μ = {mu:.3e} m²/(V·s) = {mu * 1e4:.1f} cm²/(V·s)")
    

**解答** ：μ ≈ 1.76 × 10$^{-3}$ m$^2$/(V·s) = 17.6 cm$^2$/(V·s)

### 演習1-3: van der Pauw法の計算（Medium）

Medium **問題** ：van der Pauw測定で $R_{\text{AB,CD}} = 85$ Ω、$R_{\text{BC,DA}} = 115$ Ω が得られた。シート抵抗 $R_s$ を計算し、厚さ $t = 50$ nm のときのバルク抵抗率 $\rho$ を求めよ。

**解答例を表示**
    
    
    import numpy as np
    from scipy.optimize import fsolve
    
    def van_der_pauw_eq(Rs, R1, R2):
        return np.exp(-np.pi * R1 / Rs) + np.exp(-np.pi * R2 / Rs) - 1
    
    R1 = 85  # [Ω]
    R2 = 115  # [Ω]
    
    R_s = fsolve(van_der_pauw_eq, (R1 + R2) / 2 * np.pi / np.log(2), args=(R1, R2))[0]
    
    print(f"シート抵抗 R_s = {R_s:.2f} Ω/sq")
    
    t = 50e-9  # [m]
    rho = R_s * t
    
    print(f"厚さ t = {t * 1e9:.0f} nm")
    print(f"バルク抵抗率 ρ = {rho:.3e} Ω·m = {rho * 1e8:.2f} μΩ·cm")
    

**解答** ：R$_s$ ≈ 136.8 Ω/sq、ρ ≈ 6.84 × 10$^{-6}$ Ω·m = 684 μΩ·cm

### 演習1-4: 温度依存性のフィッティング（Medium）

Medium **問題** ：金属試料の抵抗率が T = 100 K で 0.8 μΩ·cm、300 K で 2.0 μΩ·cm であった。モデル $\rho(T) = \rho_0 + AT$ でフィッティングし、$\rho_0$ と $A$ を求めよ。

**解答例を表示**
    
    
    import numpy as np
    
    T1, rho1 = 100, 0.8e-8  # [K], [Ω·m]
    T2, rho2 = 300, 2.0e-8  # [K], [Ω·m]
    
    # 線形方程式を解く
    A = (rho2 - rho1) / (T2 - T1)
    rho0 = rho1 - A * T1
    
    print(f"ρ₀ = {rho0:.3e} Ω·m = {rho0 * 1e8:.2f} μΩ·cm")
    print(f"A = {A:.3e} Ω·m/K")
    
    # 検証
    rho_100 = rho0 + A * 100
    rho_300 = rho0 + A * 300
    print(f"\n検証:")
    print(f"  ρ(100 K) = {rho_100 * 1e8:.2f} μΩ·cm（与えられた値: 0.80 μΩ·cm）")
    print(f"  ρ(300 K) = {rho_300 * 1e8:.2f} μΩ·cm（与えられた値: 2.00 μΩ·cm）")
    

**解答** ：ρ$_0$ = 2.00 × 10$^{-9}$ Ω·m = 0.20 μΩ·cm、A = 6.00 × 10$^{-11}$ Ω·m/K

### 演習1-5: TLM法の解析（Medium）

Medium **問題** ：TLM測定で、接点間距離 $L$ = 1, 2, 3, 4, 5 mm に対して、総抵抗 $R_{\text{total}}$ = 25, 30, 35, 40, 45 Ω が得られた。試料幅 $W = 1$ cm として、接触抵抗 $R_c$ とシート抵抗 $R_s$ を求めよ。

**解答例を表示**
    
    
    import numpy as np
    from scipy.stats import linregress
    
    L = np.array([1, 2, 3, 4, 5])  # [mm]
    R_total = np.array([25, 30, 35, 40, 45])  # [Ω]
    W = 0.01  # [m]
    
    slope, intercept, r_value, _, _ = linregress(L, R_total)
    
    R_c = intercept / 2
    R_s = slope * W * 1000  # W を mm に変換
    
    print(f"接触抵抗 R_c = {R_c:.2f} Ω")
    print(f"シート抵抗 R_s = {R_s:.2f} Ω/sq")
    print(f"決定係数 R² = {r_value**2:.4f}")
    

**解答** ：R$_c$ = 10.0 Ω、R$_s$ = 50.0 Ω/sq、R$^2$ = 1.0000（完璧な線形関係）

### 演習1-6: 半導体の活性化エネルギー（Hard）

Hard **問題** ：半導体試料の抵抗率が T = 300 K で 1.0 Ω·m、400 K で 0.1 Ω·m であった。モデル $\rho(T) = \rho_0 \exp(E_a / k_B T)$ でフィッティングし、活性化エネルギー $E_a$ を eV 単位で求めよ（$k_B = 8.617 \times 10^{-5}$ eV/K）。

**解答例を表示**
    
    
    import numpy as np
    
    T1, rho1 = 300, 1.0  # [K], [Ω·m]
    T2, rho2 = 400, 0.1  # [K], [Ω·m]
    kB = 8.617e-5  # [eV/K]
    
    # ln(ρ) = ln(ρ₀) + Ea/(kB T)
    # ln(rho1) = ln(rho0) + Ea/(kB T1)
    # ln(rho2) = ln(rho0) + Ea/(kB T2)
    # ln(rho1) - ln(rho2) = Ea/(kB) * (1/T1 - 1/T2)
    
    Ea = kB * (np.log(rho1) - np.log(rho2)) / (1/T1 - 1/T2)
    
    print(f"活性化エネルギー Ea = {Ea:.3f} eV")
    
    # ρ₀ を求める
    rho0 = rho1 * np.exp(-Ea / (kB * T1))
    print(f"ρ₀ = {rho0:.3e} Ω·m")
    
    # 検証
    rho_300 = rho0 * np.exp(Ea / (kB * 300))
    rho_400 = rho0 * np.exp(Ea / (kB * 400))
    print(f"\n検証:")
    print(f"  ρ(300 K) = {rho_300:.2f} Ω·m（与えられた値: 1.00 Ω·m）")
    print(f"  ρ(400 K) = {rho_400:.2f} Ω·m（与えられた値: 0.10 Ω·m）")
    

**解答** ：E$_a$ ≈ 0.661 eV、ρ$_0$ ≈ 3.94 × 10$^{-13}$ Ω·m

### 演習1-7: 四端子測定の誤差評価（Hard）

Hard **問題** ：試料抵抗 $R_{\text{sample}} = 0.5$ Ω、接触抵抗 $R_c = 10$ Ω、電流 $I = 0.1$ A のとき、二端子測定と四端子測定での相対誤差を計算せよ。また、接触抵抗が何Ω以下であれば、二端子測定でも相対誤差が5%以内に収まるか求めよ。

**解答例を表示**
    
    
    R_sample = 0.5  # [Ω]
    R_c = 10  # [Ω]
    
    # 二端子測定
    R_2terminal = 2 * R_c + R_sample
    error_relative = (R_2terminal - R_sample) / R_sample * 100
    
    print(f"試料抵抗 R_sample = {R_sample} Ω")
    print(f"接触抵抗 R_c = {R_c} Ω")
    print(f"二端子測定値: {R_2terminal} Ω")
    print(f"相対誤差: {error_relative:.1f}%")
    
    # 相対誤差5%以内の条件
    # (2*R_c + R_sample - R_sample) / R_sample <= 0.05
    # 2*R_c / R_sample <= 0.05
    # R_c <= 0.05 * R_sample / 2
    
    R_c_max = 0.05 * R_sample / 2
    print(f"\n相対誤差5%以内の条件: R_c <= {R_c_max:.4f} Ω = {R_c_max * 1000:.2f} mΩ")
    

**解答** ：相対誤差 4100%、R$_c$ ≤ 0.0125 Ω = 12.5 mΩ（非常に厳しい条件）

### 演習1-8: 実験計画（Hard）

Hard **問題** ：未知の薄膜材料（厚さ200 nm）の電気伝導特性を評価する実験計画を立案せよ。測定法（二端子/四端子、van der Pauw）、温度範囲、データ解析手法を説明せよ。

**解答例を表示**

**実験計画** ：

  1. **試料作製** ：四隅に4つの接点（直径 < 0.5 mm）を配置（van der Pauw配置）
  2. **室温測定** ： 
     * van der Pauw法で $R_{\text{AB,CD}}$ と $R_{\text{BC,DA}}$ を測定
     * シート抵抗 $R_s$ を計算、厚さ 200 nm から抵抗率 $\rho$ を算出
  3. **温度依存性測定** ： 
     * 温度範囲: 77 K（液体窒素温度）〜 400 K
     * 測定間隔: 20-30 K ごと、20-30点
     * 各温度で熱平衡を待つ（10-15分）
  4. **データ解析** ： 
     * $\rho$ vs $T$ プロットを作成
     * 金属的挙動（$\rho \propto T$）か半導体的挙動（$\rho \propto \exp(E_a / k_B T)$）かを判定
     * 適切なモデルでフィッティング（lmfit使用）
     * Matthiessenの法則で残留抵抗率 $\rho_0$ を評価
  5. **接触抵抗評価（オプション）** ：TLM法で複数の接点間距離を測定し、接触抵抗の寄与を定量化

**期待される結果** ：

  * 金属的材料: $\rho \approx 1-100$ μΩ·cm、正の温度係数
  * 半導体材料: $\rho \approx 0.1-1000$ Ω·cm、負の温度係数、活性化エネルギー 0.1-1 eV

## 1.7 学習の確認

以下のチェックリストで理解度を確認しましょう：

### 基本理解

  * Drudeモデルの仮定と電気伝導率の式 $\sigma = ne^2\tau/m$ を説明できる
  * 散乱緩和時間 $\tau$ の物理的意味を理解している
  * 二端子測定と四端子測定の原理と違いを説明できる
  * van der Pauw定理の式を理解し、シート抵抗を計算できる
  * Matthiessenの法則を理解している

### 実践スキル

  * Drudeモデルを使って電気伝導率を計算できる
  * van der Pauw法のPythonコードを実装できる
  * 温度依存性データをフィッティングできる（lmfit使用）
  * TLM法で接触抵抗を評価できる
  * 測定データの不確かさを評価できる

### 応用力

  * 金属と半導体の温度依存性の違いを説明できる
  * 実験データから散乱機構を推定できる
  * 適切な測定法（二端子/四端子、van der Pauw）を選択できる
  * 測定誤差の主要因を特定し、対策を提案できる

## 1.8 参考文献

  1. van der Pauw, L. J. (1958). _A method of measuring specific resistivity and Hall effect of discs of arbitrary shape_. Philips Research Reports, 13(1), 1-9. - van der Pauw法の原論文
  2. Drude, P. (1900). _Zur Elektronentheorie der Metalle_. Annalen der Physik, 306(3), 566-613. - Drudeモデルの原論文
  3. Schroder, D. K. (2006). _Semiconductor Material and Device Characterization_ (3rd ed.). Wiley-Interscience. - 半導体測定技術の標準テキスト
  4. Streetman, B. G., & Banerjee, S. K. (2015). _Solid State Electronic Devices_ (7th ed.). Pearson. - 電気伝導理論の教科書
  5. Ashcroft, N. W., & Mermin, N. D. (1976). _Solid State Physics_. Holt, Rinehart and Winston. - Drudeモデルと金属物性の詳細
  6. Cohen, M. H., et al. (1960). _Contact Resistance and Methods for Its Determination_. Solid-State Electronics, 1(2), 159-169. - 接触抵抗測定技術
  7. Reeves, G. K., & Harrison, H. B. (1982). _Obtaining the specific contact resistance from transmission line model measurements_. IEEE Electron Device Letters, 3(5), 111-113. - TLM法の実践的解説

## 1.9 次章へ

次章では、**Hall効果測定** の原理と実践を学びます。Hall効果は、磁場中での電荷の偏向を利用してキャリア密度とキャリア種別（電子/正孔）を決定する強力な手法です。van der Pauw法と組み合わせることで、材料の電気的特性を完全に特徴付けることができます。
