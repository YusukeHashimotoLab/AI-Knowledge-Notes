---
title: 第4章：電気的・磁気的性質
chapter_title: 第4章：電気的・磁気的性質
subtitle: 伝導現象と磁性の物理
difficulty: 中級〜上級
---

## この章で学ぶこと

### 学習目標（3レベル）

#### 基本レベル

  * Drudeモデルによる電気伝導のメカニズムを説明できる
  * Hall効果とキャリア密度・移動度の関係を理解できる
  * 強磁性・反強磁性・常磁性の違いを説明できる

#### 中級レベル

  * バンド構造から電気伝導度を計算できる
  * 磁気モーメントとスピン密度の関係を理解し、計算できる
  * スピン軌道相互作用が磁性に与える影響を説明できる

#### 上級レベル

  * DFT計算結果から電気的・磁気的性質を定量的に予測できる
  * 超伝導の基本メカニズム（BCS理論）を理解できる
  * 実験データとDFT計算を比較し、汎関数の妥当性を評価できる

## 電気伝導の古典理論：Drudeモデル

### 自由電子近似

金属中の価電子を「自由に動き回る粒子」とみなす最も単純なモデルです。電子は原子核の格子中を自由に移動し、散乱は格子振動（フォノン）や不純物によって起こります。

### Drudeモデルの基本方程式

電場$\mathbf{E}$中での電子の運動方程式：

$$ m^* \frac{d\mathbf{v}}{dt} = -e\mathbf{E} - \frac{m^*\mathbf{v}}{\tau} $$ 

  * $m^*$：電子の有効質量
  * $\mathbf{v}$：ドリフト速度
  * $\tau$：緩和時間（衝突までの平均時間）
  * $-e$：電子の電荷

定常状態（$d\mathbf{v}/dt = 0$）では：

$$ \mathbf{v} = -\frac{e\tau}{m^*}\mathbf{E} $$ 

### 電気伝導度

電流密度$\mathbf{J}$は：

$$ \mathbf{J} = -ne\mathbf{v} = \frac{ne^2\tau}{m^*}\mathbf{E} = \sigma \mathbf{E} $$ 

したがって、電気伝導度は：

$$ \sigma = \frac{ne^2\tau}{m^*} $$ 

  * $n$：キャリア密度 [m⁻³]
  * $e$：電子電荷（$1.602 \times 10^{-19}$ C）

**移動度$\mu$** との関係：

$$ \mu = \frac{e\tau}{m^*}, \quad \sigma = ne\mu $$ 

#### 典型的な値（室温）

材料 | 電気伝導度 [S/m] | キャリア密度 [m⁻³] | 移動度 [cm²/Vs]  
---|---|---|---  
Cu（銅） | 5.96 × 10⁷ | 8.5 × 10²⁸ | 43  
Si（n型） | 10³ - 10⁵ | 10²¹ - 10²³ | 1400  
GaAs（n型） | 10³ - 10⁶ | 10²¹ - 10²³ | 8500  
  
### Drudeモデルのシミュレーション
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    # 物理定数
    e = 1.602e-19  # 電子電荷 [C]
    m_e = 9.109e-31  # 電子質量 [kg]
    
    def calculate_conductivity(n, tau, m_star=1.0):
        """
        電気伝導度を計算
    
        Parameters:
        -----------
        n : float
            キャリア密度 [m^-3]
        tau : float
            緩和時間 [s]
        m_star : float
            有効質量（電子質量の単位）
    
        Returns:
        --------
        sigma : float
            電気伝導度 [S/m]
        mu : float
            移動度 [cm^2/Vs]
        """
        m_eff = m_star * m_e
        sigma = n * e**2 * tau / m_eff  # 伝導度 [S/m]
        mu = e * tau / m_eff * 1e4  # 移動度 [cm^2/Vs]
        return sigma, mu
    
    # 典型的な金属（Cu）
    n_Cu = 8.5e28  # [m^-3]
    tau_Cu = 2.7e-14  # [s]
    sigma_Cu, mu_Cu = calculate_conductivity(n_Cu, tau_Cu, m_star=1.0)
    
    print("=== 銅（Cu）の電気特性 ===")
    print(f"キャリア密度: {n_Cu:.2e} m^-3")
    print(f"緩和時間: {tau_Cu:.2e} s")
    print(f"電気伝導度: {sigma_Cu:.2e} S/m")
    print(f"移動度: {mu_Cu:.1f} cm^2/Vs")
    
    # 半導体（Si n型）の移動度と温度依存性
    temperatures = np.linspace(100, 500, 50)  # [K]
    # 移動度の温度依存性（簡略モデル: μ ∝ T^-3/2）
    mu_Si_ref = 1400  # [cm^2/Vs] at 300K
    T_ref = 300
    mu_Si = mu_Si_ref * (temperatures / T_ref)**(-1.5)
    
    plt.figure(figsize=(10, 6))
    plt.plot(temperatures, mu_Si, linewidth=2, color='#f093fb')
    plt.axhline(y=1400, color='red', linestyle='--', label='室温値 (300K)')
    plt.xlabel('温度 [K]', fontsize=12)
    plt.ylabel('移動度 [cm²/Vs]', fontsize=12)
    plt.title('Si n型半導体の移動度の温度依存性', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('mobility_temperature.png', dpi=300, bbox_inches='tight')
    plt.show()
    

## Hall効果とキャリア測定

### Hall効果の原理

電流が流れる導体に垂直な磁場をかけると、ローレンツ力により電荷が偏り、横方向に電位差（Hall電圧）が生じます。
    
    
    ```mermaid
    graph LR
        A[電流 Jx] --> B[磁場 Bz]
        B --> C[ローレンツ力F = -e v × B]
        C --> D[電荷分離]
        D --> E[Hall電圧 VH]
    
        style A fill:#f093fb,stroke:#f5576c,stroke-width:2px,color:#fff
        style E fill:#d4edda,stroke:#28a745,stroke-width:2px
    ```

### Hall係数

Hall電場$E_y$は：

$$ E_y = R_H J_x B_z $$ 

Hall係数$R_H$は：

$$ R_H = \frac{1}{ne} $$ 

  * 正孔（ホール）の場合：$R_H > 0$
  * 電子の場合：$R_H < 0$

**キャリア密度の測定** ：

$$ n = \frac{1}{|R_H| e} $$ 

**移動度の測定** ：

$$ \mu = |R_H| \sigma $$ 

### Hall効果測定のシミュレーション
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    def hall_effect_simulation(n, mu, B_range, thickness=1e-3):
        """
        Hall効果をシミュレート
    
        Parameters:
        -----------
        n : float
            キャリア密度 [m^-3]
        mu : float
            移動度 [m^2/Vs]
        B_range : array
            磁場範囲 [T]
        thickness : float
            試料厚さ [m]
        """
        e = 1.602e-19
    
        # Hall係数
        R_H = 1 / (n * e)  # [m^3/C]
    
        # 電流密度を一定と仮定
        J = 1e6  # [A/m^2]
    
        # Hall電圧
        V_H = R_H * J * B_range * thickness  # [V]
    
        # Hall抵抗
        R_Hall = V_H / (J * thickness**2)  # [Ω]
    
        return V_H, R_Hall, R_H
    
    # Si n型半導体の例
    n_Si = 1e22  # [m^-3]
    mu_Si = 0.14  # [m^2/Vs] = 1400 cm^2/Vs
    
    B_range = np.linspace(-2, 2, 100)  # [T]
    V_H, R_Hall, R_H = hall_effect_simulation(n_Si, mu_Si, B_range)
    
    # プロット
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Hall電圧 vs 磁場
    ax1.plot(B_range, V_H * 1e3, linewidth=2, color='#f093fb')
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax1.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    ax1.set_xlabel('磁場 [T]', fontsize=12)
    ax1.set_ylabel('Hall電圧 [mV]', fontsize=12)
    ax1.set_title('Hall電圧の磁場依存性', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Hall抵抗 vs 磁場
    ax2.plot(B_range, R_Hall, linewidth=2, color='#f5576c')
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.set_xlabel('磁場 [T]', fontsize=12)
    ax2.set_ylabel('Hall抵抗 [Ω]', fontsize=12)
    ax2.set_title('Hall抵抗の磁場依存性', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('hall_effect.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("=== Hall効果測定結果 ===")
    print(f"キャリア密度: {n_Si:.2e} m^-3")
    print(f"Hall係数: {R_H:.2e} m^3/C")
    print(f"Hall係数の符号: {'負（電子）' if R_H < 0 else '正（正孔）'}")
    print(f"1T磁場でのHall電圧: {V_H[np.argmin(np.abs(B_range - 1.0))] * 1e3:.3f} mV")
    

## 磁性の基礎理論

### 磁気モーメントの起源

原子・分子の磁気モーメントは、2つの寄与があります：

  1. **軌道磁気モーメント** ：電子の軌道運動による $$\mathbf{\mu}_L = -\frac{e}{2m_e}\mathbf{L}$$ 
  2. **スピン磁気モーメント** ：電子の固有角運動量（スピン）による $$\mathbf{\mu}_S = -g_s \frac{e}{2m_e}\mathbf{S}$$ 

ここで、$g_s \approx 2$はg因子です。Bohr magneton $\mu_B$を用いて：

$$ \mu_B = \frac{e\hbar}{2m_e} = 9.274 \times 10^{-24} \, \text{J/T} $$ 

### 磁化と磁化率

磁化$M$は、単位体積あたりの磁気モーメント：

$$ \mathbf{M} = \frac{1}{V}\sum_i \mathbf{\mu}_i $$ 

磁化率$\chi$は：

$$ \mathbf{M} = \chi \mathbf{H} $$ 

  * $\chi > 0$：常磁性（磁場方向に磁化）
  * $\chi < 0$：反磁性（磁場と逆方向に磁化）

### 磁性の分類

磁性 | 磁化率$\chi$ | 特徴 | 代表例  
---|---|---|---  
**反磁性** | $\chi < 0$（小） | 外部磁場に反発、温度依存性なし | Cu, Au, Si  
**常磁性** | $\chi > 0$（小） | 磁場方向に弱く磁化、Curie則（$\chi \propto 1/T$） | Al, Pt, O₂  
**強磁性** | $\chi \gg 1$ | 自発磁化、キュリー温度$T_C$以下で秩序 | Fe, Co, Ni  
**反強磁性** | $\chi > 0$（小） | 隣接スピンが反平行、Néel温度$T_N$以下で秩序 | MnO, Cr  
**フェリ磁性** | $\chi > 0$（大） | 反平行だが大きさが異なる → 正味の磁化 | Fe₃O₄（磁鉄鉱）  
  
### 強磁性の平均場理論（Weiss理論）

強磁性体では、隣接スピンが平行に揃おうとする「交換相互作用」が働きます。Weissは、各スピンが感じる「有効磁場」$H_{\text{eff}}$を導入しました：

$$ H_{\text{eff}} = H + \lambda M $$ 

$\lambda$はWeiss定数（分子場定数）です。自己無撞着方程式を解くと、キュリー温度$T_C$が得られます：

$$ T_C = \frac{C\lambda}{N_A k_B} $$ 

$T < T_C$では自発磁化が生じます。

## DFT計算による磁性の予測

### スピン分極DFT計算

磁性材料のDFT計算では、スピンアップ（↑）とスピンダウン（↓）の電子を別々に扱います（スピン分極計算）。

電子密度は、スピン成分に分解されます：

$$ n(\mathbf{r}) = n_\uparrow(\mathbf{r}) + n_\downarrow(\mathbf{r}) $$ 

**スピン密度** （磁化密度）：

$$ m(\mathbf{r}) = n_\uparrow(\mathbf{r}) - n_\downarrow(\mathbf{r}) $$ 

**磁気モーメント** ：

$$ \mu = \mu_B \int m(\mathbf{r}) d\mathbf{r} $$ 

### VASPでのスピン分極計算設定
    
    
    # VASPでスピン分極計算を設定するINCARファイル
    
    def create_magnetic_incar(system_name='Fe', initial_magmom=2.0):
        """
        磁性材料のVASP INCARファイルを生成
    
        Parameters:
        -----------
        system_name : str
            システム名
        initial_magmom : float
            初期磁気モーメント [μB/atom]
        """
    
        incar_content = f"""SYSTEM = {system_name} magnetic calculation
    
    # Electronic structure
    ENCUT = 400
    PREC = Accurate
    LREAL = Auto
    
    # Exchange-correlation
    GGA = PE
    
    # SCF convergence
    EDIFF = 1E-6
    NELM = 100
    
    # Smearing (金属のため)
    ISMEAR = 1          # Methfessel-Paxton
    SIGMA = 0.2
    
    # スピン分極計算
    ISPIN = 2           # スピン分極を有効化
    MAGMOM = {initial_magmom}  # 初期磁気モーメント [μB]
    
    # 磁気モーメントの出力
    LORBIT = 11         # 原子・軌道射影磁気モーメント
    
    # Parallelization
    NCORE = 4
    """
        return incar_content
    
    # 強磁性Fe（BCC）の計算設定
    incar_fe = create_magnetic_incar('Fe BCC', initial_magmom=2.2)
    print("=== Fe 強磁性計算 INCAR ===")
    print(incar_fe)
    
    # 反強磁性MnO（rocksalt）の計算設定
    # Mn原子の初期スピンを交互に設定
    incar_mno = """SYSTEM = MnO antiferromagnetic
    
    ENCUT = 450
    PREC = Accurate
    GGA = PE
    
    EDIFF = 1E-6
    ISMEAR = 0
    SIGMA = 0.05
    
    # スピン分極計算
    ISPIN = 2
    MAGMOM = 4.0 -4.0 4.0 -4.0 0 0 0 0  # Mn4個（交互）+ O4個（非磁性）
    
    LORBIT = 11
    NCORE = 4
    """
    
    print("\n=== MnO 反強磁性計算 INCAR ===")
    print(incar_mno)
    

### スピン密度の可視化
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    # スピン密度のダミーデータ生成（実際はVASP出力から読み込む）
    def generate_spin_density_data():
        """
        Fe原子周辺のスピン密度を模擬的に生成
        """
        x = np.linspace(-3, 3, 50)
        y = np.linspace(-3, 3, 50)
        X, Y = np.meshgrid(x, y)
    
        # ガウス分布でスピン密度を近似
        spin_density = 2.2 * np.exp(-(X**2 + Y**2) / 2)
    
        return X, Y, spin_density
    
    X, Y, spin_density = generate_spin_density_data()
    
    # 2Dプロット
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # コンター図
    contour = ax1.contourf(X, Y, spin_density, levels=20, cmap='RdBu_r')
    ax1.contour(X, Y, spin_density, levels=10, colors='black', linewidths=0.5, alpha=0.3)
    fig.colorbar(contour, ax=ax1, label='スピン密度 [μB/Å³]')
    ax1.set_xlabel('x [Å]', fontsize=12)
    ax1.set_ylabel('y [Å]', fontsize=12)
    ax1.set_title('Fe原子周辺のスピン密度（2D）', fontsize=14, fontweight='bold')
    ax1.set_aspect('equal')
    
    # 3Dサーフェス
    from matplotlib import cm
    ax2 = fig.add_subplot(122, projection='3d')
    surf = ax2.plot_surface(X, Y, spin_density, cmap=cm.coolwarm, alpha=0.8)
    ax2.set_xlabel('x [Å]', fontsize=10)
    ax2.set_ylabel('y [Å]', fontsize=10)
    ax2.set_zlabel('スピン密度 [μB/Å³]', fontsize=10)
    ax2.set_title('Fe原子周辺のスピン密度（3D）', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('spin_density.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 磁気モーメントの計算（数値積分）
    dx = X[0, 1] - X[0, 0]
    dy = Y[1, 0] - Y[0, 0]
    total_moment = np.sum(spin_density) * dx * dy
    
    print(f"\n=== 磁気モーメント計算結果 ===")
    print(f"積分磁気モーメント: {total_moment:.2f} μB")
    print(f"（実際のFe: 約2.2 μB）")
    

## スピン軌道相互作用（SOC）

### SOCの起源

電子のスピン$\mathbf{S}$と軌道角運動量$\mathbf{L}$が相互作用する効果です。相対論的効果により生じます：

$$ H_{\text{SOC}} = \lambda \mathbf{L} \cdot \mathbf{S} $$ 

$\lambda$はスピン軌道結合定数で、原子番号$Z$の増加とともに急激に大きくなります（$\lambda \propto Z^4$）。

### SOCの物理的影響

  * **磁気異方性** ：磁化の向きに依存してエネルギーが変わる
  * **磁気円二色性** （MCD）：円偏光に対する吸収の差
  * **Rashba効果** ：反転対称性の破れた系でのスピン分裂
  * **トポロジカル絶縁体** ：SOCによるバンド反転

### VASPでのSOC計算設定
    
    
    # スピン軌道相互作用を含むVASP計算設定
    
    def create_soc_incar(system_name='Pt', include_soc=True):
        """
        SOC計算のINCARファイルを生成
    
        Parameters:
        -----------
        system_name : str
            システム名
        include_soc : bool
            SOCを有効化するか
        """
    
        incar_content = f"""SYSTEM = {system_name} with SOC
    
    ENCUT = 400
    PREC = Accurate
    GGA = PE
    
    EDIFF = 1E-7        # SOC計算では高精度が必要
    ISMEAR = 1
    SIGMA = 0.2
    
    # スピン分極 + SOC
    ISPIN = 2
    """
    
        if include_soc:
            incar_content += """LSORBIT = .TRUE.    # スピン軌道相互作用を有効化
    LNONCOLLINEAR = .TRUE.  # 非共線的磁性（スピンの向きが自由）
    GGA_COMPAT = .FALSE.    # SOC計算に推奨
    """
    
        incar_content += """
    LORBIT = 11
    NCORE = 4
    """
        return incar_content
    
    # Pt（重元素、SOC重要）
    incar_pt_soc = create_soc_incar('Pt bulk', include_soc=True)
    print("=== Pt + SOC 計算 INCAR ===")
    print(incar_pt_soc)
    
    # SOCなしとの比較用
    incar_pt_no_soc = create_soc_incar('Pt bulk', include_soc=False)
    print("\n=== Pt (SOCなし) 計算 INCAR ===")
    print(incar_pt_no_soc)
    

### SOCによるバンド分裂
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    # SOCありなしのバンド構造シミュレーション
    def simulate_soc_band_splitting():
        """
        SOCによるバンド分裂を模擬的に可視化
        """
        k = np.linspace(-np.pi, np.pi, 200)
    
        # SOCなし（縮退）
        E_no_soc = np.cos(k) + 0.5 * np.cos(2*k)
    
        # SOCあり（分裂）
        lambda_soc = 0.3  # SOC強度
        E_soc_up = E_no_soc + lambda_soc * np.abs(np.sin(k))
        E_soc_down = E_no_soc - lambda_soc * np.abs(np.sin(k))
    
        return k, E_no_soc, E_soc_up, E_soc_down
    
    k, E_no_soc, E_soc_up, E_soc_down = simulate_soc_band_splitting()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # SOCなし
    ax1.plot(k/np.pi, E_no_soc, linewidth=2, color='blue', label='縮退バンド')
    ax1.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
    ax1.set_xlabel('k [π/a]', fontsize=12)
    ax1.set_ylabel('エネルギー [eV]', fontsize=12)
    ax1.set_title('SOCなし', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # SOCあり
    ax2.plot(k/np.pi, E_soc_up, linewidth=2, color='red', label='スピンアップ')
    ax2.plot(k/np.pi, E_soc_down, linewidth=2, color='blue', label='スピンダウン')
    ax2.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
    ax2.set_xlabel('k [π/a]', fontsize=12)
    ax2.set_ylabel('エネルギー [eV]', fontsize=12)
    ax2.set_title('SOCあり', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('soc_band_splitting.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # k=π/2での分裂エネルギー
    idx = len(k) // 4
    splitting = E_soc_up[idx] - E_soc_down[idx]
    print(f"\n=== SOCによるバンド分裂 ===")
    print(f"k=π/2での分裂: {splitting:.3f} eV")
    

## 超伝導の基礎

### 超伝導現象

臨界温度$T_c$以下で、電気抵抗がゼロになる現象です。1911年、Kamerlingh OnnesがHgで発見しました。

### BCS理論（1957年）

Bardeen, Cooper, Schriefferによる微視的理論。電子がフォノン（格子振動）を媒介として引力を持ち、「Cooper対」を形成します。

#### Cooper対形成のメカニズム

  1. 電子Aが格子を歪ませる（正電荷を引き寄せる）
  2. 歪んだ格子が電子Bを引き寄せる
  3. 実効的に電子A-B間に引力が働く（フォノン媒介）
  4. 反対スピン・反対運動量の電子対が形成される（$\mathbf{k}\uparrow, -\mathbf{k}\downarrow$）

**超伝導ギャップ** ：

$$ \Delta(T) = \Delta_0 \tanh\left(1.74\sqrt{\frac{T_c - T}{T}}\right) $$ 

$T=0$Kでのギャップ：

$$ \Delta_0 \approx 1.76 k_B T_c $$ 

### 代表的な超伝導体

材料 | $T_c$ [K] | 種類 | 備考  
---|---|---|---  
Hg（水銀） | 4.15 | Type I | 最初に発見された超伝導体  
Nb₃Sn | 18.3 | Type II | A15構造、磁石に使用  
YBa₂Cu₃O₇（YBCO） | 92 | 高温 | 銅酸化物、液体窒素温度以上  
MgB₂ | 39 | Type II | 単純構造、BCS理論で説明可能  
H₃S（高圧） | 203 | 高温 | 150 GPa、最高$T_c$記録  
  
### 超伝導ギャップの温度依存性
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    def superconducting_gap(T, Tc):
        """
        BCS理論による超伝導ギャップの温度依存性
    
        Parameters:
        -----------
        T : array
            温度 [K]
        Tc : float
            臨界温度 [K]
    
        Returns:
        --------
        Delta : array
            超伝導ギャップ [meV]
        """
        k_B = 8.617e-5  # Boltzmann constant [eV/K]
    
        # BCS理論の近似式
        Delta_0 = 1.76 * k_B * Tc * 1000  # [meV]
    
        Delta = np.zeros_like(T)
        mask = T < Tc
        Delta[mask] = Delta_0 * np.tanh(1.74 * np.sqrt((Tc - T[mask]) / T[mask]))
    
        return Delta
    
    # 各種超伝導体のT_c
    materials = {
        'Al': 1.2,
        'Nb': 9.2,
        'MgB₂': 39,
        'YBCO': 92
    }
    
    T = np.linspace(0.1, 100, 500)
    
    plt.figure(figsize=(10, 6))
    
    for name, Tc in materials.items():
        Delta = superconducting_gap(T, Tc)
        plt.plot(T, Delta, linewidth=2, label=f'{name} ($T_c$={Tc}K)')
    
    plt.xlabel('温度 [K]', fontsize=12)
    plt.ylabel('超伝導ギャップ Δ(T) [meV]', fontsize=12)
    plt.title('超伝導ギャップの温度依存性', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 100)
    plt.tight_layout()
    plt.savefig('superconducting_gap.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Δ_0 / k_B T_c の検証（BCS理論では1.76）
    for name, Tc in materials.items():
        k_B = 8.617e-5
        Delta_0 = 1.76 * k_B * Tc * 1000
        ratio = Delta_0 / (k_B * Tc * 1000)
        print(f"{name}: Δ₀/(kB·Tc) = {ratio:.2f}")
    

## まとめ

### この章で学んだこと

#### 電気的性質

  * Drudeモデルで電気伝導を理解：$\sigma = ne^2\tau/m^*$
  * Hall効果でキャリア密度と移動度を測定できる
  * 半導体は温度上昇で移動度が低下（$\mu \propto T^{-3/2}$）

#### 磁気的性質

  * 磁性は反磁性、常磁性、強磁性、反強磁性、フェリ磁性に分類される
  * スピン分極DFT計算（ISPIN=2）で磁気モーメントを予測できる
  * スピン軌道相互作用（SOC）は重元素で重要、磁気異方性の起源

#### 超伝導

  * BCS理論：電子がCooper対を形成し、抵抗ゼロに
  * 超伝導ギャップ：$\Delta_0 \approx 1.76 k_B T_c$
  * 高温超伝導体（YBCO: 92K）は液体窒素温度以上で動作

#### 次章への準備

  * 第5章では、光学的・熱的性質を学びます
  * 光吸収、バンドギャップ、フォノン、熱伝導を計算します

## 演習問題

#### 演習1：Drudeモデルの適用（難易度：★☆☆）

**問題** ：以下のデータから、電気伝導度と移動度を計算してください。

  * 材料: n型Si半導体
  * キャリア密度: $n = 1.0 \times 10^{22}$ m⁻³
  * 緩和時間: $\tau = 0.1$ ps = $1.0 \times 10^{-13}$ s
  * 有効質量: $m^* = 0.26 m_e$

**ヒント** ：

  * $\sigma = ne^2\tau/m^*$
  * $\mu = e\tau/m^*$
  * $e = 1.602 \times 10^{-19}$ C, $m_e = 9.109 \times 10^{-31}$ kg

**解答** ：$\sigma \approx 1.08 \times 10^4$ S/m、$\mu \approx 674$ cm²/Vs

#### 演習2：Hall効果によるキャリア判定（難易度：★★☆）

**問題** ：ある半導体に1Tの磁場を印加し、Hall測定を行ったところ、Hall係数$R_H = +5.0 \times 10^{-4}$ m³/Cでした。

  1. キャリアは電子か正孔か？
  2. キャリア密度を計算せよ
  3. 電気伝導度が$\sigma = 100$ S/mのとき、移動度を計算せよ

**解答** ：

  1. $R_H > 0$ なので正孔キャリア（p型半導体）
  2. $n = 1/(R_H \cdot e) = 1.25 \times 10^{22}$ m⁻³
  3. $\mu = R_H \cdot \sigma = 0.05$ m²/Vs = 500 cm²/Vs

#### 演習3：磁気モーメントの計算（難易度：★★☆）

**問題** ：Fe原子（BCC構造、a=2.87 Å）のスピン分極DFT計算を行い、以下の結果を得ました：

  * スピンアップ電子数: 8.1個
  * スピンダウン電子数: 5.9個

磁気モーメントを計算し、実験値（2.2 μB）と比較してください。

**ヒント** ：$\mu = (N_\uparrow - N_\downarrow) \mu_B$

**解答** ：$\mu = (8.1 - 5.9) \mu_B = 2.2 \mu_B$（実験値と一致）

#### 演習4：超伝導ギャップの計算（難易度：★★☆）

**問題** ：Nb（ニオブ）の臨界温度は$T_c = 9.2$Kです。

  1. $T = 0$Kでの超伝導ギャップ$\Delta_0$を計算せよ
  2. $T = 5$Kでの超伝導ギャップ$\Delta(5K)$を計算せよ

**ヒント** ：

  * $\Delta_0 = 1.76 k_B T_c$
  * $\Delta(T) = \Delta_0 \tanh(1.74\sqrt{(T_c - T)/T})$
  * $k_B = 8.617 \times 10^{-5}$ eV/K

**解答** ：

  1. $\Delta_0 = 1.76 \times 8.617 \times 10^{-5} \times 9.2 = 1.40$ meV
  2. $\Delta(5K) = 1.40 \times \tanh(1.74\sqrt{(9.2-5)/5}) = 1.40 \times 0.87 = 1.22$ meV

#### 演習5：VASPスピン分極計算の準備（難易度：★★★）

**問題** ：反強磁性MnO（rocksalt構造、a=4.43 Å）のスピン分極DFT計算を準備してください。

  1. ASEでMnO構造を作成（2×2×2スーパーセル）
  2. Mn原子の初期磁気モーメントを交互に設定（±5.0 μB）
  3. INCARファイルを作成（ISPIN=2, MAGMOM設定）
  4. KPOINTSファイルを作成（6×6×6メッシュ）

**評価基準** ：

  * MAGMOMがMn原子のみに設定されているか
  * Mn原子のスピンが交互配置になっているか
  * O原子の初期磁気モーメントが0に設定されているか

#### 演習6：磁化率の温度依存性（難易度：★★★）

**問題** ：常磁性物質の磁化率は、Curie則に従います：

$$ \chi = \frac{C}{T} $$ 

ここで、$C$はCurie定数です。以下のデータから、Curie定数を求めてください：

温度 [K] | 磁化率 $\chi$ [10⁻⁶]  
---|---  
100| 8.5  
200| 4.2  
300| 2.8  
400| 2.1  
  
**ヒント** ：$\chi$対$1/T$のプロットで直線の傾きがCurie定数

**解答例** ：$C \approx 8.5 \times 10^{-4}$ K（線形フィット）

## 参考文献

  1. Ashcroft, N. W., & Mermin, N. D. (1976). "Solid State Physics". Harcourt College Publishers.
  2. Kittel, C. (2004). "Introduction to Solid State Physics" (8th ed.). Wiley.
  3. Blundell, S. (2001). "Magnetism in Condensed Matter". Oxford University Press.
  4. Tinkham, M. (2004). "Introduction to Superconductivity" (2nd ed.). Dover Publications.
  5. Bardeen, J., Cooper, L. N., & Schrieffer, J. R. (1957). "Theory of Superconductivity". Physical Review, 108, 1175.
  6. VASP manual: Magnetism and SOC - https://www.vasp.at/wiki/index.php/Magnetism

### 免責事項

  * 本コンテンツは教育・研究・情報提供のみを目的としており、専門的な助言(法律・会計・技術的保証など)を提供するものではありません。
  * 本コンテンツおよび付随するCode examplesは「現状有姿(AS IS)」で提供され、明示または黙示を問わず、商品性、特定目的適合性、権利非侵害、正確性・完全性、動作・安全性等いかなる保証もしません。
  * 外部リンク、第三者が提供するデータ・ツール・ライブラリ等の内容・可用性・安全性について、作成者および東北大学は一切の責任を負いません。
  * 本コンテンツの利用・実行・解釈により直接的・間接的・付随的・特別・結果的・懲罰的損害が生じた場合でも、適用法で許容される最大限の範囲で、作成者および東北大学は責任を負いません。
  * 本コンテンツの内容は、予告なく変更・更新・提供停止されることがあります。
  * 本コンテンツの著作権・ライセンスは明記された条件(例: CC BY 4.0)に従います。当該ライセンスは通常、無保証条項を含みます。
