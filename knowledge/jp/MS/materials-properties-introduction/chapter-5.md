---
title: 第5章：光学的・熱的性質
chapter_title: 第5章：光学的・熱的性質
subtitle: 光吸収、バンドギャップ、フォノンと熱伝導
difficulty: 中級〜上級
---

## この章で学ぶこと

### 学習目標（3レベル）

#### 基本レベル

  * 光吸収のメカニズムとバンドギャップの関係を説明できる
  * フォノン（格子振動）の基本概念を理解できる
  * 熱伝導の古典論（Debyeモデル）を説明できる

#### 中級レベル

  * バンド構造から光学遷移と吸収スペクトルを計算できる
  * フォノンDOS（状態密度）をDFT計算から求められる
  * 熱伝導率と熱電特性を評価できる

#### 上級レベル

  * 誘電関数と屈折率を第一原理計算で予測できる
  * フォノン分散関係を解析し、材料の安定性を評価できる
  * 熱電変換材料の性能指数ZTを最大化する戦略を理解できる

## 光学的性質の基礎

### 光と物質の相互作用

光が材料に入射すると、以下のプロセスが起こります：

  * **反射** ：表面で光が跳ね返る
  * **吸収** ：電子が光子を吸収し、励起される
  * **透過** ：光が材料を通過する

光学特性は、**複素誘電関数** $\varepsilon(\omega)$ で記述されます：

$$ \varepsilon(\omega) = \varepsilon_1(\omega) + i\varepsilon_2(\omega) $$ 

  * $\varepsilon_1$：実部（屈折率に関係）
  * $\varepsilon_2$：虚部（吸収に関係）

### 複素屈折率

複素屈折率 $\tilde{n}$ は誘電関数と以下の関係にあります：

$$ \tilde{n}(\omega) = n(\omega) + i\kappa(\omega) $$ $$ \tilde{n}^2 = \varepsilon(\omega) $$ 

  * $n$：屈折率
  * $\kappa$：消衰係数（absorption coefficient）

実部と虚部の関係：

$$ n^2 - \kappa^2 = \varepsilon_1 $$ $$ 2n\kappa = \varepsilon_2 $$ 

### 光吸収係数

光吸収係数 $\alpha(\omega)$ は：

$$ \alpha(\omega) = \frac{2\omega\kappa(\omega)}{c} $$ 

Lambert-Beerの法則：材料中での光強度の減衰

$$ I(x) = I_0 e^{-\alpha x} $$ 

### バンドギャップと光吸収

半導体・絶縁体では、光子エネルギー $\hbar\omega$ がバンドギャップ $E_g$ を超えると、バンド間遷移により光吸収が始まります：

$$ \hbar\omega \geq E_g \quad \Rightarrow \quad \text{光吸収} $$ 

#### 直接遷移と間接遷移

  * **直接遷移** （GaAs, CdSeなど）：価電子帯頂上と伝導帯底が同じk点 → 強い光吸収
  * **間接遷移** （Si, Geなど）：異なるk点 → フォノン助力が必要、弱い光吸収

材料 | バンドギャップ [eV] | 遷移タイプ | 吸収波長 [nm]  
---|---|---|---  
Si| 1.12| 間接| ~1107  
GaAs| 1.42| 直接| ~873  
GaN| 3.4| 直接| ~365（UV）  
CdTe| 1.5| 直接| ~827  
  
### 光吸収スペクトルのシミュレーション
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    # 物理定数
    h = 4.135667e-15  # Planck constant [eV·s]
    c = 3e8  # Speed of light [m/s]
    
    def direct_transition_absorption(energy, E_g, A=1e5):
        """
        直接遷移の光吸収係数
    
        α(E) = A * sqrt(E - E_g)  for E > E_g
    
        Parameters:
        -----------
        energy : array
            光子エネルギー [eV]
        E_g : float
            バンドギャップ [eV]
        A : float
            比例定数 [cm^-1 eV^-1/2]
        """
        alpha = np.zeros_like(energy)
        mask = energy > E_g
        alpha[mask] = A * np.sqrt(energy[mask] - E_g)
        return alpha
    
    def indirect_transition_absorption(energy, E_g, A=1e4):
        """
        間接遷移の光吸収係数
    
        α(E) = A * (E - E_g)^2  for E > E_g
        """
        alpha = np.zeros_like(energy)
        mask = energy > E_g
        alpha[mask] = A * (energy[mask] - E_g)**2
        return alpha
    
    # エネルギー範囲
    energy = np.linspace(0, 4, 500)  # [eV]
    
    # GaAs（直接遷移）
    E_g_GaAs = 1.42
    alpha_GaAs = direct_transition_absorption(energy, E_g_GaAs)
    
    # Si（間接遷移）
    E_g_Si = 1.12
    alpha_Si = indirect_transition_absorption(energy, E_g_Si)
    
    # プロット
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # 吸収係数 vs エネルギー
    ax1.plot(energy, alpha_GaAs, linewidth=2, color='#f093fb', label='GaAs（直接）')
    ax1.plot(energy, alpha_Si, linewidth=2, color='#3498db', label='Si（間接）')
    ax1.axvline(x=E_g_GaAs, color='#f093fb', linestyle='--', alpha=0.5)
    ax1.axvline(x=E_g_Si, color='#3498db', linestyle='--', alpha=0.5)
    ax1.set_xlabel('光子エネルギー [eV]', fontsize=12)
    ax1.set_ylabel('吸収係数 α [cm⁻¹]', fontsize=12)
    ax1.set_title('バンド間遷移による光吸収', fontsize=14, fontweight='bold')
    ax1.set_yscale('log')
    ax1.set_ylim(1e2, 1e6)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 波長換算
    wavelength = 1240 / energy  # λ[nm] = 1240 / E[eV]
    ax2.plot(wavelength, alpha_GaAs, linewidth=2, color='#f093fb', label='GaAs')
    ax2.plot(wavelength, alpha_Si, linewidth=2, color='#3498db', label='Si')
    ax2.set_xlabel('波長 [nm]', fontsize=12)
    ax2.set_ylabel('吸収係数 α [cm⁻¹]', fontsize=12)
    ax2.set_title('波長依存性', fontsize=14, fontweight='bold')
    ax2.set_xlim(300, 1200)
    ax2.set_yscale('log')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('optical_absorption.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("=== バンドギャップと吸収波長 ===")
    print(f"GaAs: E_g = {E_g_GaAs} eV, λ_edge = {1240/E_g_GaAs:.0f} nm")
    print(f"Si:   E_g = {E_g_Si} eV, λ_edge = {1240/E_g_Si:.0f} nm")
    

## フォノン（格子振動）

### フォノンとは

結晶中の原子は平衡位置の周りで振動しています。この集団的な振動の量子化された励起を**フォノン** と呼びます。

$N$個の原子からなる結晶には、$3N$個の振動モードがあります：

  * **音響フォノン** （3モード）：長波長で音波として伝播（LA, TA1, TA2）
  * **光学フォノン** （$3N-3$モード）：原子が逆位相で振動、赤外活性

### フォノン分散関係

フォノンの角周波数 $\omega$ と波数 $\mathbf{q}$ の関係を**分散関係** $\omega(\mathbf{q})$ と呼びます。

1次元単原子鎖の分散関係：

$$ \omega(q) = \sqrt{\frac{4K}{M}} \left|\sin\left(\frac{qa}{2}\right)\right| $$ 

  * $K$：バネ定数（原子間力定数）
  * $M$：原子質量
  * $a$：格子定数

### フォノン状態密度（DOS）

フォノンDOS $g(\omega)$ は、角周波数 $\omega$ のフォノン状態の数を表します：

$$ g(\omega) = \frac{1}{(2\pi)^3} \int \delta(\omega - \omega_s(\mathbf{q})) d\mathbf{q} $$ 

$s$はバンドインデックス（音響/光学モード）です。

### フォノン分散関係のシミュレーション
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    def phonon_dispersion_1D_monoatomic(q, K=10, M=1, a=1):
        """
        1次元単原子鎖のフォノン分散
    
        Parameters:
        -----------
        q : array
            波数 [rad/m]
        K : float
            バネ定数 [N/m]
        M : float
            原子質量 [kg]
        a : float
            格子定数 [m]
        """
        omega = np.sqrt(4 * K / M) * np.abs(np.sin(q * a / 2))
        return omega
    
    def phonon_dispersion_1D_diatomic(q, K=10, M1=1, M2=2, a=1):
        """
        1次元二原子鎖のフォノン分散（光学・音響ブランチ）
    
        Returns:
        --------
        omega_acoustic : array
            音響フォノン
        omega_optical : array
            光学フォノン
        """
        # 簡略化した分散関係
        omega_max = np.sqrt(2 * K * (1/M1 + 1/M2))
        omega_min = 0
    
        # 音響ブランチ
        omega_acoustic = omega_max / 2 * np.abs(np.sin(q * a / 2))
    
        # 光学ブランチ
        omega_optical = omega_max * np.sqrt(1 - 0.5 * (np.sin(q * a / 2))**2)
    
        return omega_acoustic, omega_optical
    
    # 波数範囲（第一ブリルアンゾーン）
    q = np.linspace(-np.pi, np.pi, 200)
    
    # 単原子鎖
    omega_mono = phonon_dispersion_1D_monoatomic(q)
    
    # 二原子鎖
    omega_ac, omega_op = phonon_dispersion_1D_diatomic(q)
    
    # プロット
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # 単原子鎖
    ax1.plot(q/np.pi, omega_mono, linewidth=2, color='#f093fb')
    ax1.set_xlabel('波数 q [π/a]', fontsize=12)
    ax1.set_ylabel('角周波数 ω [rad/s]', fontsize=12)
    ax1.set_title('1次元単原子鎖のフォノン分散', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-1, 1)
    
    # 二原子鎖
    ax2.plot(q/np.pi, omega_ac, linewidth=2, color='#3498db', label='音響ブランチ')
    ax2.plot(q/np.pi, omega_op, linewidth=2, color='#f5576c', label='光学ブランチ')
    ax2.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
    ax2.set_xlabel('波数 q [π/a]', fontsize=12)
    ax2.set_ylabel('角周波数 ω [rad/s]', fontsize=12)
    ax2.set_title('1次元二原子鎖のフォノン分散', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(-1, 1)
    
    plt.tight_layout()
    plt.savefig('phonon_dispersion.png', dpi=300, bbox_inches='tight')
    plt.show()
    

## DFTによるフォノン計算

### DFPT（Density Functional Perturbation Theory）

フォノンを計算するには、原子を変位させたときのエネルギー変化（力定数）を求める必要があります。DFPT（密度汎関数摂動理論）を使うと、効率的に計算できます。

**力定数行列** ：

$$ \Phi_{\alpha\beta}(ll') = \frac{\partial^2 E}{\partial u_\alpha(l) \partial u_\beta(l')} $$ 

  * $u_\alpha(l)$：原子 $l$ の $\alpha$ 方向変位

**フォノン周波数** ：

$$ \omega^2(\mathbf{q}) = \text{eigenvalues of } D(\mathbf{q}) $$ 

動力学行列 $D(\mathbf{q})$ は力定数行列のフーリエ変換です。

### VASPでのフォノン計算設定
    
    
    # VASPでフォノン計算を行うINCARファイル
    
    def create_phonon_incar(system_name='Si'):
        """
        フォノン計算のINCARファイル生成
    
        VASP単体ではDFPTは限定的。Phonopy等の外部ツールを推奨。
        """
        incar_content = f"""SYSTEM = {system_name} phonon calculation
    
    # Electronic structure
    ENCUT = 500         # 高精度が必要
    PREC = Accurate
    EDIFF = 1E-8        # 非常に高精度な収束
    
    # SCF
    ISMEAR = 0
    SIGMA = 0.01        # 小さいスメアリング
    
    # Force calculation
    IBRION = -1         # 一点計算（Phonopyが変位を管理）
    NSW = 0
    
    # High precision forces
    ADDGRID = .TRUE.    # 高精度グリッド
    LREAL = .FALSE.     # 実空間射影をオフ（精度優先）
    
    NCORE = 4
    """
        return incar_content
    
    incar_phonon = create_phonon_incar('Si')
    print("=== フォノン計算 INCAR（Phonopy併用） ===")
    print(incar_phonon)
    
    # Phonopyワークフロー
    print("\n=== Phonopyワークフロー ===")
    workflow = """
    1. スーパーセル作成:
       phonopy -d --dim="2 2 2"
    
    2. 変位構造でVASP計算:
       各 POSCAR-XXX に対して VASP実行
    
    3. 力定数計算:
       phonopy -f vasprun-001.xml vasprun-002.xml ...
    
    4. フォノンバンド・DOS計算:
       phonopy --dim="2 2 2" -p band.conf
       phonopy --dim="2 2 2" -p mesh.conf
    
    5. 熱物性計算（比熱、エントロピー）:
       phonopy --dim="2 2 2" -t -p mesh.conf
    """
    print(workflow)
    

### フォノンDOSの可視化
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    # フォノンDOSのダミーデータ生成（実際はPhonopy出力から読み込む）
    def generate_phonon_dos():
        """
        Siのフォノンdos（模擬データ）
        """
        freq = np.linspace(0, 600, 300)  # [cm^-1]
    
        # 音響フォノン（低周波）
        dos_acoustic = 2 * np.exp(-(freq - 150)**2 / (2 * 50**2))
    
        # 光学フォノン（高周波）
        dos_optical = 1.5 * np.exp(-(freq - 520)**2 / (2 * 30**2))
    
        dos_total = dos_acoustic + dos_optical
    
        return freq, dos_total, dos_acoustic, dos_optical
    
    freq, dos_total, dos_acoustic, dos_optical = generate_phonon_dos()
    
    # プロット
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # フォノンDOS
    ax1.plot(freq, dos_total, linewidth=2, color='black', label='Total')
    ax1.fill_between(freq, 0, dos_acoustic, alpha=0.5, color='#3498db', label='Acoustic')
    ax1.fill_between(freq, dos_acoustic, dos_total, alpha=0.5, color='#f5576c', label='Optical')
    ax1.set_xlabel('周波数 [cm⁻¹]', fontsize=12)
    ax1.set_ylabel('フォノンDOS [states/cm⁻¹]', fontsize=12)
    ax1.set_title('Si フォノン状態密度', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 周波数 → エネルギー換算
    energy_meV = freq * 0.124  # 1 cm^-1 ≈ 0.124 meV
    ax2.plot(energy_meV, dos_total, linewidth=2, color='black')
    ax2.fill_between(energy_meV, 0, dos_acoustic, alpha=0.5, color='#3498db')
    ax2.fill_between(energy_meV, dos_acoustic, dos_total, alpha=0.5, color='#f5576c')
    ax2.set_xlabel('エネルギー [meV]', fontsize=12)
    ax2.set_ylabel('フォノンDOS', fontsize=12)
    ax2.set_title('エネルギー換算', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('phonon_dos.png', dpi=300, bbox_inches='tight')
    plt.show()
    

## 熱的性質

### 熱容量

フォノンDOSから、定積熱容量 $C_V$ を計算できます（Einstein/Debye モデル）：

$$ C_V = k_B \int g(\omega) \left(\frac{\hbar\omega}{k_B T}\right)^2 \frac{e^{\hbar\omega/k_B T}}{(e^{\hbar\omega/k_B T} - 1)^2} d\omega $$ 

**Dulong-Petitの法則** （高温極限）：

$$ C_V \to 3Nk_B $$ 

### 熱伝導

熱伝導率 $\kappa$ は、フォノンによる熱輸送で決まります：

$$ \kappa = \frac{1}{3} C_V v_s^2 \tau $$ 

  * $C_V$：熱容量
  * $v_s$：音速（フォノン群速度）
  * $\tau$：フォノンの緩和時間（散乱時間）

#### 熱伝導率の代表値（室温）

  * ダイヤモンド：2,200 W/(m·K)（最高クラス）
  * Cu（銅）：400 W/(m·K)
  * Si：150 W/(m·K)
  * ステンレス鋼：15 W/(m·K)
  * 空気：0.026 W/(m·K)

### 熱伝導率のシミュレーション
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    def thermal_conductivity_kinetic(C_V, v_s, tau):
        """
        運動論による熱伝導率
    
        κ = (1/3) C_V v_s^2 τ
    
        Parameters:
        -----------
        C_V : float
            体積熱容量 [J/(m^3·K)]
        v_s : float
            音速 [m/s]
        tau : float
            緩和時間 [s]
        """
        kappa = (1/3) * C_V * v_s**2 * tau
        return kappa
    
    # 各種材料のパラメータ
    materials = {
        'Si': {'C_V': 1.66e6, 'v_s': 8433, 'tau': 4e-11},
        'Ge': {'C_V': 1.70e6, 'v_s': 5400, 'tau': 3e-11},
        'GaAs': {'C_V': 1.50e6, 'v_s': 5150, 'tau': 2e-11},
    }
    
    print("=== 熱伝導率計算 ===")
    for name, params in materials.items():
        kappa = thermal_conductivity_kinetic(**params)
        print(f"{name}: κ = {kappa:.1f} W/(m·K)")
    
    # 温度依存性（フォノン散乱）
    temperatures = np.linspace(100, 600, 100)  # [K]
    
    # 高温: τ ∝ 1/T (Umklapp散乱)
    tau_ref = 4e-11  # 300K
    T_ref = 300
    tau_T = tau_ref * (temperatures / T_ref)**(-1)
    
    # 熱容量（Debyeモデル近似）
    C_V_300 = 1.66e6
    C_V_T = C_V_300 * (temperatures / T_ref)**3 / (np.exp(temperatures / T_ref) - 1)
    
    # 熱伝導率
    v_s = 8433
    kappa_T = (1/3) * C_V_T * v_s**2 * tau_T
    
    # プロット
    plt.figure(figsize=(10, 6))
    plt.plot(temperatures, kappa_T, linewidth=2, color='#f093fb')
    plt.axvline(x=300, color='red', linestyle='--', label='室温')
    plt.xlabel('温度 [K]', fontsize=12)
    plt.ylabel('熱伝導率 κ [W/(m·K)]', fontsize=12)
    plt.title('Si 熱伝導率の温度依存性', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('thermal_conductivity_temperature.png', dpi=300, bbox_inches='tight')
    plt.show()
    

## 熱電変換材料

### Seebeck効果

温度差 $\Delta T$ が電位差 $\Delta V$ を生む現象：

$$ \Delta V = S \Delta T $$ 

$S$は**Seebeck係数** （熱電能）です。

### 熱電性能指数 ZT

熱電変換材料の性能は、**無次元性能指数 ZT** で評価されます：

$$ ZT = \frac{S^2 \sigma T}{\kappa} $$ 

  * $S$：Seebeck係数 [V/K]
  * $\sigma$：電気伝導度 [S/m]
  * $\kappa$：熱伝導率 [W/(m·K)]
  * $T$：絶対温度 [K]

**高性能熱電材料の条件** ：

  * 高Seebeck係数（$S > 200$ μV/K）
  * 高電気伝導度（金属並み）
  * 低熱伝導率（ガラス並み）
  * → 「電子結晶・フォノンガラス」（Phonon Glass Electron Crystal, PGEC）

材料 | ZT（室温） | ZT（最適温度） | 用途  
---|---|---|---  
Bi₂Te₃| 0.8-1.0| 1.0（300K）| 冷却素子  
PbTe| 0.5| 1.5（700K）| 発電  
Half-Heusler（TiNiSn）| 0.5| 1.0（800K）| 高温発電  
SnSe（単結晶）| 0.5| 2.6（923K）| 研究中  
  
### ZTの最適化シミュレーション
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    def thermoelectric_ZT(S, sigma, kappa, T):
        """
        熱電性能指数ZTを計算
    
        Parameters:
        -----------
        S : float
            Seebeck係数 [V/K]
        sigma : float
            電気伝導度 [S/m]
        kappa : float
            熱伝導率 [W/(m·K)]
        T : float
            温度 [K]
        """
        ZT = (S**2 * sigma * T) / kappa
        return ZT
    
    # キャリア密度の変化に伴うS, σ, κの変化（簡略モデル）
    carrier_density = np.logspace(18, 22, 100)  # [m^-3]
    
    # Seebeck係数（キャリア密度が増えると減少）
    S = 300e-6 / (carrier_density / 1e20)**0.5  # [V/K]
    
    # 電気伝導度（キャリア密度に比例）
    sigma = 1e3 * (carrier_density / 1e20)  # [S/m]
    
    # 熱伝導率（電子とフォノンの寄与）
    kappa_lattice = 1.5  # 格子熱伝導 [W/(m·K)]
    kappa_electronic = 2.44e-8 * sigma * 300  # Wiedemann-Franz則
    kappa = kappa_lattice + kappa_electronic
    
    # ZT計算
    T = 300  # [K]
    ZT = thermoelectric_ZT(S, sigma, kappa, T)
    
    # プロット
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    # Seebeck係数
    ax1.plot(carrier_density, S * 1e6, linewidth=2, color='#f093fb')
    ax1.set_xscale('log')
    ax1.set_xlabel('キャリア密度 [m⁻³]', fontsize=12)
    ax1.set_ylabel('Seebeck係数 [μV/K]', fontsize=12)
    ax1.set_title('Seebeck係数', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # 電気伝導度
    ax2.plot(carrier_density, sigma, linewidth=2, color='#3498db')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlabel('キャリア密度 [m⁻³]', fontsize=12)
    ax2.set_ylabel('電気伝導度 [S/m]', fontsize=12)
    ax2.set_title('電気伝導度', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # 熱伝導率
    ax3.plot(carrier_density, kappa, linewidth=2, color='#f5576c', label='Total')
    ax3.axhline(y=kappa_lattice, color='green', linestyle='--', label='Lattice')
    ax3.set_xscale('log')
    ax3.set_xlabel('キャリア密度 [m⁻³]', fontsize=12)
    ax3.set_ylabel('熱伝導率 [W/(m·K)]', fontsize=12)
    ax3.set_title('熱伝導率', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # ZT
    ax4.plot(carrier_density, ZT, linewidth=2, color='black')
    optimal_idx = np.argmax(ZT)
    ax4.plot(carrier_density[optimal_idx], ZT[optimal_idx], 'ro', markersize=10, label=f'Max ZT = {ZT[optimal_idx]:.2f}')
    ax4.set_xscale('log')
    ax4.set_xlabel('キャリア密度 [m⁻³]', fontsize=12)
    ax4.set_ylabel('ZT', fontsize=12)
    ax4.set_title('熱電性能指数ZT', fontsize=14, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('thermoelectric_ZT_optimization.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n=== 最適ZT ===")
    print(f"最適キャリア密度: {carrier_density[optimal_idx]:.2e} m^-3")
    print(f"最大ZT: {ZT[optimal_idx]:.2f}")
    print(f"対応するSeebeck係数: {S[optimal_idx]*1e6:.1f} μV/K")
    print(f"対応する電気伝導度: {sigma[optimal_idx]:.1f} S/m")
    

## DFT計算による光学・熱物性の予測

### 誘電関数の計算
    
    
    # VASPで光学特性（誘電関数）を計算するINCARファイル
    
    def create_optics_incar(system_name='Si', nbands=None):
        """
        光学計算のINCARファイル生成
    
        Parameters:
        -----------
        system_name : str
            システム名
        nbands : int
            バンド数（デフォルトの2倍程度推奨）
        """
        nbands_str = f"NBANDS = {nbands}" if nbands else "# NBANDS = 自動設定の2倍推奨"
    
        incar_content = f"""SYSTEM = {system_name} optical properties
    
    ENCUT = 400
    PREC = Accurate
    GGA = PE
    
    EDIFF = 1E-6
    ISMEAR = 0
    SIGMA = 0.05
    
    # 光学計算
    LOPTICS = .TRUE.    # 誘電関数計算を有効化
    {nbands_str}
    
    # 高密度k点が必要
    # KPOINTSで16×16×16以上推奨
    
    NCORE = 4
    """
        return incar_content
    
    incar_optics = create_optics_incar('Si', nbands=48)
    print("=== 光学計算 INCAR ===")
    print(incar_optics)
    
    # 計算される物理量
    print("\n=== 計算される光学物性 ===")
    print("- 複素誘電関数 ε(ω) = ε₁(ω) + iε₂(ω)")
    print("- 吸収係数 α(ω)")
    print("- 屈折率 n(ω)")
    print("- 反射率 R(ω)")
    print("\n出力ファイル: vasprun.xml （タグ）")
    

### フォノンによる熱物性の計算
    
    
    # Phonopyで熱物性を計算
    
    phonopy_workflow = """
    === Phonopyによる熱物性計算ワークフロー ===
    
    1. フォノン計算（前述）:
       phonopy --dim="2 2 2" -c POSCAR -f vasprun-*.xml
    
    2. メッシュ設定ファイル作成（mesh.conf）:
       DIM = 2 2 2
       MP = 16 16 16
       TPROP = .TRUE.
       TMIN = 0
       TMAX = 1000
       TSTEP = 10
    
    3. 熱物性計算:
       phonopy -t mesh.conf --dim="2 2 2"
    
    4. 出力ファイル:
       - thermal_properties.yaml
         * 定積熱容量 C_V(T)
         * エントロピー S(T)
         * 自由エネルギー F(T)
    
    5. プロット:
       phonopy -t -p mesh.conf
    """
    
    print(phonopy_workflow)
    
    # 熱物性の温度依存性（Debyeモデルによる近似）
    def debye_heat_capacity(T, T_D, N=1):
        """
        Debyeモデルによる熱容量
    
        Parameters:
        -----------
        T : array
            温度 [K]
        T_D : float
            Debye温度 [K]
        N : int
            原子数
        """
        k_B = 1.380649e-23  # Boltzmann constant [J/K]
        x = T_D / T
    
        def debye_function(x):
            # 数値積分が必要（簡略化）
            return 3 * (x / np.sinh(x/2))**2
    
        C_V = 3 * N * k_B * debye_function(x)
        return C_V
    
    # Si のDebye温度: 645 K
    T = np.linspace(10, 600, 100)
    C_V = debye_heat_capacity(T, T_D=645, N=1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(T, C_V / 1.380649e-23, linewidth=2, color='#f093fb')
    plt.axhline(y=3, color='red', linestyle='--', label='Dulong-Petit則（高温極限）')
    plt.xlabel('温度 [K]', fontsize=12)
    plt.ylabel('熱容量 C_V / k_B', fontsize=12)
    plt.title('Si 熱容量の温度依存性（Debyeモデル）', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('heat_capacity_debye.png', dpi=300, bbox_inches='tight')
    plt.show()
    

## まとめ

### この章で学んだこと

#### 光学的性質

  * 光吸収は複素誘電関数 $\varepsilon(\omega)$ で記述される
  * バンドギャップ以上の光子エネルギーで強い吸収が起こる
  * 直接遷移（GaAs）は間接遷移（Si）より強い光吸収を示す
  * DFT計算で誘電関数、屈折率、吸収係数を予測できる

#### フォノンと熱的性質

  * フォノンは格子振動の量子化された励起
  * 音響フォノンと光学フォノンに分類される
  * フォノンDOSから熱容量、エントロピーを計算できる
  * 熱伝導率はフォノン輸送で決まる：$\kappa = (1/3) C_V v_s^2 \tau$

#### 熱電変換

  * 性能指数 $ZT = S^2\sigma T / \kappa$ で評価
  * 高ZTには高S、高σ、低κが必要（PGEC概念）
  * キャリア密度の最適化が重要

#### 次章への準備

  * 第6章では、これまで学んだ全てを統合した実践ワークフローを学びます
  * Si、GaN、Fe、BaTiO₃を例に、構造最適化→DFT→物性解析の全プロセスを実行します

## 演習問題

#### 演習1：光吸収の計算（難易度：★☆☆）

**問題** ：GaN（バンドギャップ 3.4 eV）の吸収端波長を計算してください。

**ヒント** ：$\lambda = hc / E = 1240 / E[eV]$ [nm]

**解答** ：$\lambda = 1240 / 3.4 = 365$ nm（紫外領域）

#### 演習2：フォノン分散の解釈（難易度：★★☆）

**問題** ：1次元単原子鎖のフォノン分散 $\omega(q) = \omega_{\max} |\sin(qa/2)|$ について：

  1. $q=0$（長波長極限）での周波数は？物理的意味は？
  2. $q=\pi/a$（ブリルアンゾーン境界）での周波数は？
  3. 群速度 $v_g = d\omega/dq$ を $q=0$ で計算せよ

**解答** ：

  1. $\omega(0) = 0$（並進運動、音波として伝播）
  2. $\omega(\pi/a) = \omega_{\max}$（隣接原子が逆位相）
  3. $v_g = d\omega/dq|_{q=0} = (\omega_{\max}a/2) \cos(0) = \omega_{\max}a/2$（音速）

#### 演習3：熱伝導率の推定（難易度：★★☆）

**問題** ：以下のデータからSiの熱伝導率を推定してください。

  * 体積熱容量：$C_V = 1.66 \times 10^6$ J/(m³·K)
  * 音速：$v_s = 8433$ m/s
  * フォノン緩和時間：$\tau = 4 \times 10^{-11}$ s

**ヒント** ：$\kappa = (1/3) C_V v_s^2 \tau$

**解答** ：$\kappa = (1/3) \times 1.66 \times 10^6 \times 8433^2 \times 4 \times 10^{-11} = 157$ W/(m·K)

#### 演習4：ZTの最適化（難易度：★★★）

**問題** ：熱電材料のZTを最大化するために、以下のパラメータをどう変更すべきか説明してください。

  1. Seebeck係数 $S$ を2倍にする
  2. 電気伝導度 $\sigma$ を2倍にする
  3. 熱伝導率 $\kappa$ を1/2にする

**推奨解答** ：

  * $S$ を2倍 → ZT は4倍（$ZT \propto S^2$）
  * $\sigma$ を2倍 → ZT は2倍（$ZT \propto \sigma$）
  * $\kappa$ を1/2 → ZT は2倍（$ZT \propto 1/\kappa$）
  * 結論：Seebeck係数の改善が最も効果的。ただし、S, σ, κは相互依存するため、単独制御は困難。キャリア密度の最適化が現実的。

#### 演習5：Phonopy計算の準備（難易度：★★★）

**問題** ：Si結晶のフォノン計算をPhonopyで実行する準備をしてください。

  1. 2×2×2スーパーセルを作成するコマンドを書け
  2. 変位構造の数を予測せよ（対称性考慮なし）
  3. 各変位構造に対するVASP計算のINCARパラメータを設定せよ

**解答例** ：

  1. `phonopy -d --dim="2 2 2"`
  2. Si（ダイヤモンド構造）: 2原子/unitcell × 8 unitcells = 16原子。各原子3方向変位 → 変位構造数はPhonopyが対称性で削減（通常1-2個）
  3. INCAR: IBRION=-1, EDIFF=1E-8, ADDGRID=.TRUE., LREAL=.FALSE.（高精度力計算）

#### 演習6：実践課題（難易度：★★★）

**問題** ：GaAsの光学特性を予測するDFT計算を設計してください。

  1. バンドギャップ計算に適した汎関数を選択せよ（理由も）
  2. LOPTICS=.TRUEで計算する際の推奨k点メッシュとNBANDSを示せ
  3. 計算結果から吸収スペクトルを抽出する方法を説明せよ

**推奨解答** ：

  1. HSE06汎関数（GGA-PBEはバンドギャップを過小評価）
  2. k点: 16×16×16（Γ中心）、NBANDS: デフォルトの2倍（伝導帯の高エネルギー状態も含める）
  3. vasprun.xmlからを読み、ε₂(ω)から吸収係数α(ω)を計算：$\alpha = 2\omega\kappa/c$、$\kappa$はε₂から導出

## 参考文献

  1. Fox, M. (2010). "Optical Properties of Solids" (2nd ed.). Oxford University Press.
  2. Dove, M. T. (1993). "Introduction to Lattice Dynamics". Cambridge University Press.
  3. Togo, A., & Tanaka, I. (2015). "First principles phonon calculations in materials science". Scripta Materialia, 108, 1-5.
  4. Snyder, G. J., & Toberer, E. S. (2008). "Complex thermoelectric materials". Nature Materials, 7, 105-114.
  5. Ashcroft & Mermin (1976). "Solid State Physics". Chapters on phonons and thermal properties.
  6. Phonopy documentation: https://phonopy.github.io/phonopy/

### 免責事項

  * 本コンテンツは教育・研究・情報提供のみを目的としており、専門的な助言(法律・会計・技術的保証など)を提供するものではありません。
  * 本コンテンツおよび付随するCode examplesは「現状有姿(AS IS)」で提供され、明示または黙示を問わず、商品性、特定目的適合性、権利非侵害、正確性・完全性、動作・安全性等いかなる保証もしません。
  * 外部リンク、第三者が提供するデータ・ツール・ライブラリ等の内容・可用性・安全性について、作成者および東北大学は一切の責任を負いません。
  * 本コンテンツの利用・実行・解釈により直接的・間接的・付随的・特別・結果的・懲罰的損害が生じた場合でも、適用法で許容される最大限の範囲で、作成者および東北大学は責任を負いません。
  * 本コンテンツの内容は、予告なく変更・更新・提供停止されることがあります。
  * 本コンテンツの著作権・ライセンスは明記された条件(例: CC BY 4.0)に従います。当該ライセンスは通常、無保証条項を含みます。
