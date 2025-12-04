---
title: 第1章：固体電子論の基礎
chapter_title: 第1章：固体電子論の基礎
subtitle: バンド理論入門 - なぜ金属は導電性があり、絶縁体にはないのか
---

🌐 JP | [🇬🇧 EN](<../../../en/MS/materials-properties-introduction/chapter-1.html>) | Last sync: 2025-11-16

[AI寺子屋トップ](<../../index.html>)›[材料科学](<../../MS/index.html>)›[Materials Properties](<../../MS/materials-properties-introduction/index.html>)›Chapter 1

## 学習目標

この章を完了すると、以下を説明できるようになります：

### 基本理解

  * 自由電子モデルとフェルミエネルギーの概念
  * バンド構造の形成メカニズムと禁制帯の起源
  * 金属、半導体、絶縁体の電子構造の違い

### 実践スキル

  * Pythonでフェルミエネルギーと状態密度を計算できる
  * 簡単なバンド構造図をプロットし解釈できる
  * フェルミ面の概念を視覚化し理解できる

### 応用力

  * 実験データからバンドギャップを推定できる
  * 材料の電気的性質をバンド理論で予測できる
  * 新規材料の電子構造を設計できる基礎知識を持つ

## 1.1 なぜバンド理論が必要なのか

### 1.1.1 日常の疑問から始めよう

銅線は電気を通すのに、なぜゴムは電気を通さないのでしょうか？どちらも原子から構成されているのに、この違いはどこから来るのでしょう。

この単純な疑問に答えるために、物理学者たちは20世紀前半に「バンド理論」を確立しました。この理論は材料科学の最も重要な基礎理論の一つであり、以下を説明できます：

  * **導電性の起源** : なぜ金属は電気を通し、絶縁体は通さないのか
  * **半導体の動作原理** : トランジスタや太陽電池がどう機能するのか
  * **光学的性質** : なぜダイヤモンドは透明で銅は不透明なのか
  * **磁性** : 材料が磁石になる条件は何か

**💡 歴史的背景**

バンド理論の基礎は1920年代にFelix BlochとRudolf Peierlsによって築かれました。1930年代にはWilson、Brillouin、Wannierらがこれを発展させ、固体の電子構造を記述する普遍的な枠組みを確立しました。この理論は半導体革命（トランジスタの発明、1947年）を支える理論的基盤となりました。

### 1.1.2 古典論の限界

古典物理学（19世紀）では、電子を自由に動き回る粒子として扱うDrudeモデル（1900年）が提案されました。このモデルは導電性をある程度説明できましたが、以下の重大な問題がありました：

現象 | Drudeモデルの予測 | 実験結果  
---|---|---  
比熱への電子の寄与 | 大きい（3/2 Nk_B） | 非常に小さい（~0.01 Nk_B）  
絶縁体の存在 | 説明不可 | 明確に存在  
半導体の温度依存性 | 金属と同じ | 全く異なる（指数関数的）  
  
これらの矛盾を解決するには、**量子力学** と**結晶の周期的構造** を考慮する必要があることが明らかになりました。

## 1.2 自由電子モデル - 最も単純な量子論的描像

### 1.2.1 量子力学の基本: 粒子は波である

量子力学の基本原理（de Broglie、1924）によれば、すべての粒子は波としての性質を持ちます：

λ = h / p  
（波長 = プランク定数 / 運動量） 

電子のような軽い粒子では、この波長が原子間距離（~数Å）と同程度になるため、**波としての性質が顕著に現れます** 。

### 1.2.2 箱の中の電子: 量子化されたエネルギー

1次元の「箱」（長さ L）に閉じ込められた電子を考えます。シュレーディンガー方程式を解くと、エネルギーは離散的な値しか取れません：

E_n = (n²π²ℏ²) / (2m L²) （n = 1, 2, 3, ...） 

ここで重要なポイント：

  * エネルギーは連続ではなく**離散的** （量子化）
  * 最低エネルギー（n=1）でもゼロではない（ゼロ点エネルギー）
  * 箱が大きいほど（L↑）、エネルギー間隔は狭くなる（E_n+1 - E_n ∝ 1/L²）

### 1.2.3 3次元への拡張: 実際の金属

現実の金属を体積 V = L³ の箱と考えると、電子の状態は3つの量子数 (n_x, n_y, n_z) で指定されます：

E = (π²ℏ²/2mL²) × (n_x² + n_y² + n_z²) 

この式は、エネルギー E を持つ状態が**3次元の n-空間で半径 R = √(n_x² + n_y² + n_z²) の球面上** に分布することを意味します。

### 1.2.4 フェルミエネルギー - 電子の海の水位

金属中の電子は**フェルミ粒子** （パウリの排他原理に従う）であり、1つの量子状態には最大2個（スピン↑↓）しか入れません。

N 個の電子が最低エネルギー状態から順に詰まっていくと、絶対零度（T=0K）で最も高いエネルギーを持つ電子のエネルギーを**フェルミエネルギー E_F** と呼びます：

E_F = (ℏ²/2m) × (3π²n)^(2/3) 

ここで n = N/V は電子密度です。

**✅ 重要な数値例**

**銅（Cu）** の場合：

  * 電子密度: n ≈ 8.45 × 10²² cm⁻³
  * フェルミエネルギー: E_F ≈ 7.0 eV
  * フェルミ温度: T_F = E_F/k_B ≈ 81,000 K

室温（300K）は T_F の 0.4% にすぎません。これが**電子の比熱が小さい理由** です。熱的に励起される電子は全体のごく一部（~T/T_F）だけなのです。

### 1.2.5 状態密度 (Density of States, DOS)

エネルギー E から E+dE の範囲に何個の量子状態があるかを示す関数を**状態密度 D(E)** といいます。自由電子モデルでは：

D(E) = (V/2π²) × (2m/ℏ²)^(3/2) × √E 

重要な特徴：

  * D(E) ∝ √E : エネルギーの平方根に比例
  * 低エネルギーでは状態が少ない
  * 高エネルギーになるほど状態が増える

**💡 Pro Tip**

状態密度は材料の物性を理解する上で極めて重要です。電気伝導度、比熱、磁化率など、多くの物理量は D(E_F)（フェルミエネルギーでの状態密度）に比例します。

## 1.3 Pythonで計算してみよう - フェルミエネルギーと状態密度

### Example 1: 基本計算 - フェルミエネルギー
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy import constants
    
    # 物理定数
    hbar = constants.hbar  # 換算プランク定数 (J·s)
    m_e = constants.m_e    # 電子質量 (kg)
    e = constants.e        # 電気素量 (C)
    
    def fermi_energy(n):
        """
        自由電子モデルでのフェルミエネルギーを計算
    
        Args:
            n (float): 電子密度 [m^-3]
    
        Returns:
            float: フェルミエネルギー [eV]
        """
        E_F = (hbar**2 / (2 * m_e)) * (3 * np.pi**2 * n)**(2/3)
        return E_F / e  # JをeVに変換
    
    # 典型的な金属の電子密度
    metals = {
        'Li': 4.70e28,   # リチウム
        'Na': 2.65e28,   # ナトリウム
        'Cu': 8.45e28,   # 銅
        'Ag': 5.85e28,   # 銀
        'Au': 5.90e28    # 金
    }
    
    print("金属のフェルミエネルギー")
    print("-" * 40)
    for metal, n in metals.items():
        E_F = fermi_energy(n)
        T_F = E_F * e / constants.k  # フェルミ温度 [K]
        print(f"{metal:3s}: E_F = {E_F:5.2f} eV, T_F = {T_F/1000:5.1f} × 10³ K")
    
    # 出力例:
    # 金属のフェルミエネルギー
    # ----------------------------------------
    # Li : E_F =  4.74 eV, T_F =  55.0 × 10³ K
    # Na : E_F =  3.24 eV, T_F =  37.6 × 10³ K
    # Cu : E_F =  7.00 eV, T_F =  81.2 × 10³ K
    # Ag : E_F =  5.49 eV, T_F =  63.7 × 10³ K
    # Au : E_F =  5.53 eV, T_F =  64.2 × 10³ K
    

### Example 2: 状態密度の可視化
    
    
    def density_of_states(E, V):
        """
        3次元自由電子の状態密度
    
        Args:
            E (array): エネルギー [J]
            V (float): 体積 [m^3]
    
        Returns:
            array: 状態密度 [J^-1 m^-3]
        """
        factor = V / (2 * np.pi**2) * (2 * m_e / hbar**2)**(3/2)
        DOS = factor * np.sqrt(E)
        return DOS
    
    # 銅のパラメータ
    n_Cu = 8.45e28  # m^-3
    E_F_Cu = fermi_energy(n_Cu) * e  # J
    V = 1e-6  # 1 mm³
    
    # エネルギー範囲（0から2E_Fまで）
    E = np.linspace(0.01 * E_F_Cu, 2 * E_F_Cu, 1000)
    DOS = density_of_states(E, V)
    
    # プロット
    plt.figure(figsize=(10, 6))
    plt.plot(E / e, DOS * e, 'b-', linewidth=2, label='D(E)')
    plt.axvline(E_F_Cu / e, color='r', linestyle='--', linewidth=2, label=f'E_F = {E_F_Cu/e:.2f} eV')
    plt.fill_between(E / e, 0, DOS * e, where=(E <= E_F_Cu), alpha=0.3, color='blue', label='占有状態 (T=0K)')
    
    plt.xlabel('Energy (eV)', fontsize=12)
    plt.ylabel('Density of States (eV⁻¹ mm⁻³)', fontsize=12)
    plt.title('3D Free Electron Model: Density of States', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # 絶対零度での全電子数を検証
    occupied_DOS = DOS[E <= E_F_Cu]
    dE = E[1] - E[0]
    N_electrons = 2 * np.sum(occupied_DOS) * dE  # 因子2はスピン
    print(f"計算された電子数: {N_electrons:.2e}")
    print(f"期待値 (n×V): {n_Cu * V:.2e}")
    # 両者は一致するはず
    

### Example 3: フェルミ-ディラック分布と温度効果
    
    
    def fermi_dirac(E, E_F, T):
        """
        フェルミ-ディラック分布関数
    
        Args:
            E (array): エネルギー [J]
            E_F (float): フェルミエネルギー [J]
            T (float): 温度 [K]
    
        Returns:
            array: 占有確率 [0, 1]
        """
        if T == 0:
            return (E <= E_F).astype(float)
        else:
            k_B = constants.k
            return 1 / (1 + np.exp((E - E_F) / (k_B * T)))
    
    # 銅の場合
    E = np.linspace(0, 2 * E_F_Cu, 1000)
    temperatures = [0, 300, 1000, 3000, 10000]  # K
    
    plt.figure(figsize=(12, 5))
    
    # (a) フェルミ-ディラック分布
    plt.subplot(1, 2, 1)
    for T in temperatures:
        f = fermi_dirac(E, E_F_Cu, T)
        label = f'T = {T} K' if T > 0 else 'T = 0 K'
        plt.plot(E / e, f, linewidth=2, label=label)
    
    plt.xlabel('Energy (eV)', fontsize=12)
    plt.ylabel('Occupation Probability f(E)', fontsize=12)
    plt.title('Fermi-Dirac Distribution', fontsize=14, fontweight='bold')
    plt.axvline(E_F_Cu / e, color='k', linestyle='--', alpha=0.5)
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)
    
    # (b) 有効的に励起される電子（T > 0で変化する部分）
    plt.subplot(1, 2, 2)
    E_range = np.linspace(E_F_Cu - 0.5*e, E_F_Cu + 0.5*e, 500)
    for T in [300, 1000, 3000]:
        f = fermi_dirac(E_range, E_F_Cu, T)
        thermal_width = 2 * constants.k * T / e  # ~k_B T のエネルギー幅
        plt.plot(E_range / e, f, linewidth=2, label=f'T = {T} K (Δ ≈ {thermal_width*1000:.1f} meV)')
    
    plt.xlabel('Energy (eV)', fontsize=12)
    plt.ylabel('Occupation Probability f(E)', fontsize=12)
    plt.title('Thermal Broadening near E_F', fontsize=14, fontweight='bold')
    plt.axvline(E_F_Cu / e, color='k', linestyle='--', alpha=0.5)
    plt.axhline(0.5, color='gray', linestyle=':', alpha=0.5)
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # 室温での励起される電子の割合
    T_room = 300  # K
    thermal_energy = constants.k * T_room
    fraction = T_room / (E_F_Cu / constants.k)
    print(f"\n室温（{T_room} K）での解析:")
    print(f"熱エネルギー k_B T ≈ {thermal_energy/e*1000:.1f} meV")
    print(f"フェルミエネルギー E_F = {E_F_Cu/e:.2f} eV")
    print(f"励起される電子の割合: ~{fraction*100:.2f}%")
    

**⚠️ 注意点**

自由電子モデルは多くの定性的予測に成功しますが、以下は説明できません：

  * **絶縁体の存在** : なぜ電子があっても電気を通さない物質があるのか
  * **半導体のバンドギャップ** : 禁制帯はどこから来るのか
  * **有効質量の変化** : なぜ材料によって電子の質量が変わるのか

これらを理解するには、**結晶の周期的ポテンシャル** を考慮する必要があります。

## 1.4 バンド構造の形成 - 周期的ポテンシャルの効果

### 1.4.1 なぜバンド（帯）が形成されるのか

結晶中の電子は、規則正しく配列した原子核が作る**周期的なポテンシャル** 中を運動します。この周期性が電子のエネルギー構造に劇的な変化をもたらします。

**💡 直感的理解**

孤立した原子では電子は離散的なエネルギー準位（1s, 2s, 2p, ...）を持ちます。N個の原子が集まって結晶を作ると：

  1. 各準位がN個に分裂（相互作用による）
  2. N ~ 10²³ と非常に大きいため、準位は実質的に連続的な「帯」（バンド）を形成
  3. しかし、異なる原子軌道由来のバンドの間には**禁制帯（バンドギャップ）** が残る

### 1.4.2 Blochの定理

周期的ポテンシャル V(r + R) = V(r) 中のシュレーディンガー方程式の解は、**Bloch関数** という特別な形を取ります：

ψ_nk(r) = e^(ik·r) u_nk(r) 

ここで：

  * k: **波数ベクトル** （電子の結晶運動量に対応、ℏk）
  * u_nk(r): 結晶の周期性を持つ関数（u_nk(r + R) = u_nk(r)）
  * n: **バンド指数** （1s由来、2p由来、など）

エネルギーはkとnに依存します：**E = E_n(k)**

### 1.4.3 ブリルアンゾーンと分散関係

1次元の単純な例（格子定数 a）を考えると、波数 k は**第一ブリルアンゾーン** -π/a ≤ k ≤ π/a の範囲で物理的に独立です。

**✅ 重要な概念**

**ブリルアンゾーンの境界（k = ±π/a）で何が起こるか？**

自由電子では E ∝ k² と滑らかに変化しますが、周期ポテンシャルがあると：

  * k = π/a で**エネルギーギャップ** が開く
  * 電子波が結晶格子で**Bragg反射** を起こすため
  * このギャップが**バンドギャップ** の起源

### 1.4.4 簡単なモデル: 1次元結晶のバンド構造

最も単純な「ほぼ自由電子近似」では、周期ポテンシャルを弱い摂動として扱うと、分散関係は次のように変化します：

E_±(k) = (ℏ²k²)/(2m) ± |V_G| × √(1 + (ℏ²k·G)/(m|V_G|)²) 

ここで V_G は周期ポテンシャルのフーリエ成分、G は逆格子ベクトルです。

### Example 4: 1次元バンド構造のプロット
    
    
    def band_structure_1d(k, a, V_G):
        """
        1次元ほぼ自由電子モデルのバンド構造
    
        Args:
            k (array): 波数 [m^-1]
            a (float): 格子定数 [m]
            V_G (float): 周期ポテンシャル [J]
    
        Returns:
            tuple: (E_lower, E_upper) 下側と上側のバンド [J]
        """
        E_free = (hbar * k)**2 / (2 * m_e)  # 自由電子
    
        # ブリルアンゾーン境界でのギャップ
        G = 2 * np.pi / a
        gap = 2 * V_G
    
        # 摂動を含むバンド構造（簡略化）
        cos_term = np.cos(k * a)
        E_lower = E_free - V_G * np.abs(cos_term)
        E_upper = E_free + V_G * np.abs(cos_term)
    
        return E_lower, E_upper
    
    # パラメータ
    a = 3e-10  # 3 Å の格子定数
    V_G = 2 * e  # 2 eV のポテンシャル
    
    # 第一ブリルアンゾーン
    k = np.linspace(-np.pi/a, np.pi/a, 500)
    E_lower, E_upper = band_structure_1d(k, a, V_G)
    
    # 自由電子（比較用）
    E_free = (hbar * k)**2 / (2 * m_e)
    
    # プロット
    plt.figure(figsize=(10, 7))
    plt.plot(k * a / np.pi, E_free / e, 'k--', linewidth=1.5, alpha=0.5, label='Free electron')
    plt.plot(k * a / np.pi, E_lower / e, 'b-', linewidth=2.5, label='Lower band')
    plt.plot(k * a / np.pi, E_upper / e, 'r-', linewidth=2.5, label='Upper band')
    
    # バンドギャップ領域を強調
    k_gap = np.pi / a
    E_gap_lower = band_structure_1d(np.array([k_gap]), a, V_G)[0][0] / e
    E_gap_upper = band_structure_1d(np.array([k_gap]), a, V_G)[1][0] / e
    plt.fill_between([-1, 1], E_gap_lower, E_gap_upper, alpha=0.2, color='gray', label='Band gap')
    
    plt.xlabel('Wave vector k (π/a)', fontsize=12)
    plt.ylabel('Energy (eV)', fontsize=12)
    plt.title('1D Band Structure: Nearly Free Electron Model', fontsize=14, fontweight='bold')
    plt.axhline(0, color='k', linewidth=0.5)
    plt.axvline(-1, color='k', linewidth=0.5, linestyle=':')
    plt.axvline(1, color='k', linewidth=0.5, linestyle=':')
    plt.text(0.7, E_gap_upper + 0.5, f'Band gap\n~{(E_gap_upper - E_gap_lower):.2f} eV', fontsize=10, ha='center')
    plt.legend(fontsize=11, loc='upper right')
    plt.grid(alpha=0.3)
    plt.xlim(-1, 1)
    plt.tight_layout()
    plt.show()
    

## 1.5 状態密度（DOS）再訪 - バンド構造からの計算

### 1.5.1 状態密度の一般的定義

バンド構造 E_n(k) が与えられたとき、エネルギー E での状態密度は：

D(E) = Σ_n ∫ δ(E - E_n(k)) (dk/(2π)³) 

実際の計算では、等エネルギー面 E_n(k) = E の面積を計算します。

### 1.5.2 バンド構造がもたらすDOSの特徴

自由電子の D(E) ∝ √E とは異なり、バンド構造を持つ系では：

  * **バンドギャップ** : D(E) = 0 の領域（禁制帯）
  * **Van Hove特異点** : ∇_k E = 0 となる点で D(E) が発散的に増加
  * **バンド端** : バンドの最高点・最低点で D(E) の特徴的な変化

### Example 5: バンド構造からの状態密度計算
    
    
    def dos_from_band(E_array, E_band, V):
        """
        バンド構造から状態密度を計算（ヒストグラム法）
    
        Args:
            E_array (array): エネルギー軸 [J]
            E_band (array): バンドのエネルギー [J]（1Dの場合はk点の配列）
            V (float): 体積 [m^3]
    
        Returns:
            array: 状態密度 [J^-1 m^-3]
        """
        hist, bin_edges = np.histogram(E_band, bins=E_array, density=False)
        dE = E_array[1] - E_array[0]
        # 1Dの場合の規格化（実際には3Dで異なる）
        DOS = hist / (dE * V) * 2  # スピン因子2
        return DOS
    
    # 1次元バンド構造の例
    k = np.linspace(-np.pi/a, np.pi/a, 10000)
    E_lower, E_upper = band_structure_1d(k, a, V_G)
    
    # エネルギー軸
    E_axis = np.linspace(-2*e, 20*e, 500)
    
    # 各バンドのDOSを計算
    DOS_lower = dos_from_band(E_axis, E_lower, 1e-9)
    DOS_upper = dos_from_band(E_axis, E_upper, 1e-9)
    DOS_total = DOS_lower + DOS_upper
    
    # プロット
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # (a) バンド構造
    ax1.plot(k * a / np.pi, E_lower / e, 'b-', linewidth=2.5, label='Lower band')
    ax1.plot(k * a / np.pi, E_upper / e, 'r-', linewidth=2.5, label='Upper band')
    ax1.set_xlabel('Wave vector k (π/a)', fontsize=12)
    ax1.set_ylabel('Energy (eV)', fontsize=12)
    ax1.set_title('(a) Band Structure', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(alpha=0.3)
    ax1.set_xlim(-1, 1)
    
    # (b) 状態密度
    E_plot = E_axis / e
    ax2.plot(DOS_total * e, E_plot, 'k-', linewidth=2.5, label='Total DOS')
    ax2.fill_betweenx(E_plot, 0, DOS_total * e, alpha=0.3, color='blue')
    ax2.set_xlabel('Density of States (arb. units)', fontsize=12)
    ax2.set_ylabel('Energy (eV)', fontsize=12)
    ax2.set_title('(b) Density of States', fontsize=13, fontweight='bold')
    ax2.axhline(E_gap_lower, color='gray', linestyle='--', alpha=0.5)
    ax2.axhline(E_gap_upper, color='gray', linestyle='--', alpha=0.5)
    ax2.text(0.7 * np.max(DOS_total * e), (E_gap_lower + E_gap_upper) / 2,
             'Band gap\n(D=0)', fontsize=10, ha='center', va='center',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax2.legend(fontsize=11)
    ax2.grid(alpha=0.3)
    ax2.set_ylim(E_plot[0], E_plot[-1])
    
    plt.tight_layout()
    plt.show()
    

## 1.6 金属・半導体・絶縁体の分類

### 1.6.1 バンドの占有状態で決まる電気的性質

材料の導電性は、**フェルミエネルギー E_F がバンド構造のどこに位置するか** で決まります。

材料タイプ | バンド占有 | バンドギャップ | 導電性（室温） | 典型例  
---|---|---|---|---  
**金属** | 価電子帯が部分的に占有  
または複数バンドが重複 | なし（または負） | 10⁴-10⁶ S/cm | Cu, Al, Na  
**半導体** | 価電子帯が完全に占有  
伝導帯は空 | 小さい（0.1-3 eV） | 10⁻⁶-10² S/cm | Si, GaAs, GaN  
**絶縁体** | 価電子帯が完全に占有  
伝導帯は空 | 大きい（>3 eV） | <10⁻¹⁰ S/cm | SiO₂, Al₂O₃  
  
### 1.6.2 なぜ金属は電気を通すのか

金属では：

  1. E_F がバンドの**途中** に位置
  2. E_F 近傍に**占有された状態と空の状態が共存**
  3. 電場をかけると、E_F 付近の電子が容易に空の状態へ移動（=電流）
  4. D(E_F) > 0 であることが本質

**💡 具体例: ナトリウム（Na）**

Na原子は電子配置 [Ne] 3s¹ を持ちます。結晶中では：

  * 3sバンドが形成されるが、各原子が1個の3s電子しか持たないため**バンドは半分しか埋まらない**
  * したがって E_F はバンド中央に位置
  * → 優れた金属伝導性（σ ≈ 2.1 × 10⁵ S/cm）

### 1.6.3 なぜ絶縁体は電気を通さないのか

絶縁体（例: ダイヤモンド）では：

  1. 価電子がちょうど価電子帯を**完全に埋める**
  2. E_F は価電子帯の上端と伝導帯の下端の**中間（バンドギャップ内）**
  3. バンドギャップが大きい（ダイヤモンド: 5.5 eV）
  4. 室温の熱エネルギー（~0.026 eV）では電子を伝導帯に励起できない
  5. → 導電性なし（σ < 10⁻¹⁴ S/cm）

### 1.6.4 半導体 - 中間の性質

半導体は絶縁体と似た構造ですが、バンドギャップが小さい（Si: 1.1 eV, GaAs: 1.4 eV）ため：

  * **真性伝導** : 熱励起で少数の電子が伝導帯へ（キャリア密度 ∝ exp(-E_g/2k_BT)）
  * **不純物伝導** : ドーピングで制御可能なキャリア密度
  * **温度依存性** : 温度上昇で導電性が指数関数的に増加（金属は逆）

**✅ 数値例: Siの真性キャリア密度**

シリコン（T = 300 K）：

  * バンドギャップ: E_g = 1.12 eV
  * 真性キャリア密度: n_i ≈ 1.5 × 10¹⁰ cm⁻³
  * 比較: Si原子密度 5 × 10²² cm⁻³ → 1兆個に1個だけ励起

ドーピング（例: 10¹⁶ cm⁻³ のリン）により、キャリア密度を100万倍に増やせます。

### Example 6: 金属・半導体・絶縁体のバンド構造比較
    
    
    def plot_band_occupation():
        """
        金属・半導体・絶縁体のバンド占有状態を可視化
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    
        E_range = np.linspace(-3, 3, 100)
    
        # (a) 金属
        ax = axes[0]
        # 重なったバンド
        ax.fill_between([-1, 1], -2, 0, alpha=0.6, color='blue', label='Valence band (filled)')
        ax.fill_between([-1, 1], 0, 2, alpha=0.3, color='blue', label='Conduction band (partial)')
        ax.axhline(0.5, color='red', linewidth=2, linestyle='--', label='Fermi level (E_F)')
        ax.text(0, 0.7, 'E_F', fontsize=12, color='red', ha='center', fontweight='bold')
        ax.set_ylim(-3, 3)
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylabel('Energy (eV)', fontsize=12)
        ax.set_title('(a) Metal', fontsize=13, fontweight='bold')
        ax.legend(fontsize=9, loc='upper right')
        ax.grid(alpha=0.3)
        ax.set_xticks([])
    
        # (b) 半導体
        ax = axes[1]
        ax.fill_between([-1, 1], -2, -0.5, alpha=0.6, color='blue', label='Valence band (filled)')
        ax.fill_between([-1, 1], 0.5, 2, alpha=0.3, color='lightblue', label='Conduction band (empty)')
        ax.axhline(0, color='red', linewidth=2, linestyle='--', label='Fermi level (E_F)')
        ax.text(0, 0.15, 'E_F', fontsize=12, color='red', ha='center', fontweight='bold')
        # バンドギャップ
        ax.annotate('', xy=(1.2, -0.5), xytext=(1.2, 0.5),
                    arrowprops=dict(arrowstyle='<->', color='black', lw=2))
        ax.text(1.5, 0, 'E_g\n~1 eV', fontsize=11, va='center', ha='left')
        ax.set_ylim(-3, 3)
        ax.set_xlim(-1.5, 1.8)
        ax.set_title('(b) Semiconductor', fontsize=13, fontweight='bold')
        ax.legend(fontsize=9, loc='upper right')
        ax.grid(alpha=0.3)
        ax.set_xticks([])
        ax.set_yticks([])
    
        # (c) 絶縁体
        ax = axes[2]
        ax.fill_between([-1, 1], -2, -1, alpha=0.6, color='blue', label='Valence band (filled)')
        ax.fill_between([-1, 1], 2, 3, alpha=0.3, color='lightblue', label='Conduction band (empty)')
        ax.axhline(0.5, color='red', linewidth=2, linestyle='--', label='Fermi level (E_F)')
        ax.text(0, 0.7, 'E_F', fontsize=12, color='red', ha='center', fontweight='bold')
        # バンドギャップ
        ax.annotate('', xy=(1.2, -1), xytext=(1.2, 2),
                    arrowprops=dict(arrowstyle='<->', color='black', lw=2))
        ax.text(1.5, 0.5, 'E_g\n>3 eV', fontsize=11, va='center', ha='left')
        ax.set_ylim(-3, 3.5)
        ax.set_xlim(-1.5, 1.8)
        ax.set_title('(c) Insulator', fontsize=13, fontweight='bold')
        ax.legend(fontsize=9, loc='upper right')
        ax.grid(alpha=0.3)
        ax.set_xticks([])
        ax.set_yticks([])
    
        plt.tight_layout()
        plt.show()
    
    plot_band_occupation()
    

## 1.7 フェルミ面 - 3次元バンド構造の理解

### 1.7.1 フェルミ面とは

3次元の**k空間** （波数空間、逆格子空間とも呼ぶ）で、E_n(k) = E_F を満たす点の集合を**フェルミ面** と呼びます。

なぜ重要か：

  * 電気伝導、磁性、超伝導など、多くの物性はフェルミ面近傍の電子が担う
  * フェルミ面の**形状（トポロジー）** が物性を決定
  * 実験的に測定可能（de Haas-van Alphen効果、ARPES等）

### 1.7.2 自由電子のフェルミ面 - 完全な球

自由電子モデルでは E = ℏ²k²/(2m) なので、E = E_F を満たすkは：

|k| = k_F = √(2mE_F)/ℏ = (3π²n)^(1/3) 

つまり、フェルミ面は**半径 k_F の球** です（フェルミ球）。

### 1.7.3 実際の金属のフェルミ面 - 複雑な形状

結晶の周期ポテンシャルにより、フェルミ面は球から歪みます：

  * **銅（Cu）** : ほぼ球形だが、ブリルアンゾーン境界で「首」が形成
  * **金（Au）** : 立方体の角に突起を持つ複雑な形状
  * **鉄（Fe）** : 磁性と関連した複雑なマルチシート構造

### Example 7: 2次元フェルミ面の可視化
    
    
    def fermi_surface_2d():
        """
        2次元の簡単なバンド構造からフェルミ面を計算
        """
        # 2次元グリッド
        kx = np.linspace(-np.pi/a, np.pi/a, 200)
        ky = np.linspace(-np.pi/a, np.pi/a, 200)
        KX, KY = np.meshgrid(kx, ky)
    
        # 簡単なバンド構造（tight-binding model）
        t = 2 * e  # hopping parameter
        E = -2 * t * (np.cos(KX * a) + np.cos(KY * a))
    
        # いくつかのフェルミエネルギーでのフェルミ面
        E_F_values = [-3*t, -2*t, 0, 2*t]
    
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        axes = axes.flatten()
    
        for idx, E_F in enumerate(E_F_values):
            ax = axes[idx]
    
            # バンド構造のコンター
            contour = ax.contourf(KX*a/np.pi, KY*a/np.pi, E/t, levels=50, cmap='RdBu_r', alpha=0.7)
    
            # フェルミ面（E = E_F の等高線）
            ax.contour(KX*a/np.pi, KY*a/np.pi, E/t, levels=[E_F/t], colors='black', linewidths=3)
    
            # ブリルアンゾーン境界
            ax.plot([-1, 1, 1, -1, -1], [-1, -1, 1, 1, -1], 'k--', linewidth=2, alpha=0.5)
    
            ax.set_xlabel('k_x (π/a)', fontsize=11)
            ax.set_ylabel('k_y (π/a)', fontsize=11)
            ax.set_title(f'E_F = {E_F/t:.1f}t', fontsize=12, fontweight='bold')
            ax.set_aspect('equal')
            ax.grid(alpha=0.3)
    
            # カラーバー
            plt.colorbar(contour, ax=ax, label='E/t')
    
        plt.suptitle('Fermi Surface Evolution in 2D Tight-Binding Model',
                     fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout()
        plt.show()
    
    fermi_surface_2d()
    

### Example 8: 3次元フェルミ面（簡単な例）
    
    
    from mpl_toolkits.mplot3d import Axes3D
    
    def fermi_surface_3d_sphere():
        """
        自由電子のフェルミ面（球）を3Dプロット
        """
        # 銅の例
        n_Cu = 8.45e28  # m^-3
        k_F = (3 * np.pi**2 * n_Cu)**(1/3)
    
        # 球面パラメータ
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        kx = k_F * np.outer(np.cos(u), np.sin(v))
        ky = k_F * np.outer(np.sin(u), np.sin(v))
        kz = k_F * np.outer(np.ones(np.size(u)), np.cos(v))
    
        # プロット
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
    
        ax.plot_surface(kx * 1e-10, ky * 1e-10, kz * 1e-10,
                        color='cyan', alpha=0.6, edgecolor='navy', linewidth=0.1)
    
        # ブリルアンゾーン境界（立方体）
        G = np.pi / a
        # 立方体のエッジを描画
        for i in [-1, 1]:
            for j in [-1, 1]:
                ax.plot([i*G*1e-10, i*G*1e-10], [j*G*1e-10, j*G*1e-10],
                       [-G*1e-10, G*1e-10], 'k--', alpha=0.3)
                ax.plot([i*G*1e-10, i*G*1e-10], [-G*1e-10, G*1e-10],
                       [j*G*1e-10, j*G*1e-10], 'k--', alpha=0.3)
                ax.plot([-G*1e-10, G*1e-10], [i*G*1e-10, i*G*1e-10],
                       [j*G*1e-10, j*G*1e-10], 'k--', alpha=0.3)
    
        ax.set_xlabel('k_x (Å⁻¹)', fontsize=12)
        ax.set_ylabel('k_y (Å⁻¹)', fontsize=12)
        ax.set_zlabel('k_z (Å⁻¹)', fontsize=12)
        ax.set_title('Fermi Surface of Copper (Free Electron Approximation)',
                     fontsize=14, fontweight='bold')
    
        # 等方的なアスペクト比
        max_range = G * 1e-10
        ax.set_xlim([-max_range, max_range])
        ax.set_ylim([-max_range, max_range])
        ax.set_zlim([-max_range, max_range])
    
        plt.tight_layout()
        plt.show()
    
        print(f"銅のフェルミ波数: k_F = {k_F:.2e} m⁻¹ = {k_F*1e-10:.2f} Å⁻¹")
    
    fermi_surface_3d_sphere()
    

### 1.7.4 フェルミ面と輸送現象

フェルミ面の幾何学的性質が物性に直結します：

フェルミ面の特徴 | 物理的影響 | 典型例  
---|---|---  
球形に近い | 等方的な導電性 | Cu, Al  
楕円体 | 異方的な導電性 | Bi（結晶方向で100倍の差）  
ネスティング  
（並行な部分） | 電荷密度波、超伝導 | 層状化合物  
トポロジカルな変化 | Lifshitz転移 | 圧力下の金属  
  
## 1.8 実験的検証 - バンド理論は正しいか？

### 1.8.1 角度分解光電子分光（ARPES）

最も直接的にバンド構造を測定できる手法：

  * 光を照射して試料から電子を叩き出す（光電効果）
  * 放出された電子のエネルギーと角度を測定
  * エネルギー保存則から E(k) を直接決定

**✅ 測定精度（最新のARPES）**

  * エネルギー分解能: ~1 meV
  * 角度分解能: ~0.1°
  * 時間分解ARPES: フェムト秒（10⁻¹⁵秒）スケール

これにより、超伝導ギャップ（~1 meV）や電荷秩序の形成過程まで観測可能。

### 1.8.2 光学測定 - バンドギャップの決定

半導体のバンドギャップは光学吸収で精密に測定できます：

α(ω) ∝ √(ℏω - E_g) （直接遷移の場合） 

ここで α は吸収係数、ω は光の角振動数です。

### Example 9: 光学吸収からのバンドギャップ推定
    
    
    def optical_absorption(photon_energy, E_g, A=1.0, indirect=False):
        """
        半導体の光学吸収係数を計算
    
        Args:
            photon_energy (array): 光子エネルギー [eV]
            E_g (float): バンドギャップ [eV]
            A (float): 比例定数
            indirect (bool): 間接遷移の場合True
    
        Returns:
            array: 吸収係数 [任意単位]
        """
        alpha = np.zeros_like(photon_energy)
        above_gap = photon_energy > E_g
    
        if indirect:
            # 間接遷移: α ∝ (ℏω - E_g)²
            alpha[above_gap] = A * (photon_energy[above_gap] - E_g)**2
        else:
            # 直接遷移: α ∝ √(ℏω - E_g)
            alpha[above_gap] = A * np.sqrt(photon_energy[above_gap] - E_g)
    
        return alpha
    
    # 典型的な半導体のパラメータ
    semiconductors = {
        'GaN': {'E_g': 3.4, 'type': 'direct'},
        'Si': {'E_g': 1.12, 'type': 'indirect'},
        'GaAs': {'E_g': 1.42, 'type': 'direct'},
        'Ge': {'E_g': 0.66, 'type': 'indirect'}
    }
    
    # プロット
    photon_energy = np.linspace(0, 4, 500)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, (name, params) in enumerate(semiconductors.items()):
        ax = axes[idx]
        E_g = params['E_g']
        indirect = (params['type'] == 'indirect')
    
        alpha = optical_absorption(photon_energy, E_g, A=1.0, indirect=indirect)
    
        ax.plot(photon_energy, alpha, 'b-', linewidth=2.5)
        ax.axvline(E_g, color='r', linestyle='--', linewidth=2, label=f'E_g = {E_g:.2f} eV')
        ax.fill_between(photon_energy, 0, alpha, where=(photon_energy > E_g), alpha=0.3, color='blue')
    
        ax.set_xlabel('Photon Energy (eV)', fontsize=11)
        ax.set_ylabel('Absorption Coefficient α (arb. units)', fontsize=11)
        ax.set_title(f'{name} ({params["type"]} gap)', fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)
        ax.set_xlim(0, 4)
        ax.set_ylim(0, np.max(alpha) * 1.1)
    
    plt.suptitle('Optical Absorption Spectra of Semiconductors', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    # Tauc plotによるバンドギャップ決定（実験データ解析法）
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for name, params in semiconductors.items():
        E_g = params['E_g']
        indirect = (params['type'] == 'indirect')
        alpha = optical_absorption(photon_energy, E_g, A=1.0, indirect=indirect)
    
        # Tauc plot: (αhν)^(1/n) vs hν （n=1/2 for direct, n=2 for indirect）
        if indirect:
            y_axis = (alpha * photon_energy)**0.5  # (αhν)^1/2
        else:
            y_axis = (alpha * photon_energy)**2  # (αhν)^2
    
        ax.plot(photon_energy, y_axis, linewidth=2, label=name)
        ax.axvline(E_g, linestyle=':', alpha=0.5)
    
    ax.set_xlabel('Photon Energy (eV)', fontsize=12)
    ax.set_ylabel('(αhν)^n (arb. units)', fontsize=12)
    ax.set_title('Tauc Plot for Band Gap Determination', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 4)
    plt.tight_layout()
    plt.show()
    
    print("\n💡 Tauc plotの使い方:")
    print("グラフの立ち上がり部分を直線外挿してx軸との交点を求めると、")
    print("それがバンドギャップ E_g に対応します。")
    

### 1.8.3 量子振動 - フェルミ面の精密測定

**de Haas-van Alphen効果** は、強磁場中で磁化が振動する現象で、フェルミ面の断面積を精密に測定できます：

振動周期 ∝ 1/A  
（A: 磁場方向に垂直なフェルミ面の極値断面積） 

この方法で、フェルミ面のトポロジー、有効質量、散乱率などを決定できます。

## 学習目標の確認

このchapterを完了したあなたは、以下を説明できるようになりました：

### 基本理解

  * ✅ 自由電子モデルとフェルミエネルギーの概念
  * ✅ バンド構造の形成メカニズムと禁制帯の起源
  * ✅ 金属、半導体、絶縁体の電子構造の違い

### 実践スキル

  * ✅ Pythonでフェルミエネルギーと状態密度を計算できる
  * ✅ 簡単なバンド構造図をプロットし解釈できる
  * ✅ フェルミ面の概念を視覚化し理解できる

### 応用力

  * ✅ 実験データからバンドギャップを推定できる
  * ✅ 材料の電気的性質をバンド理論で予測できる
  * ✅ 新規材料の電子構造を設計できる基礎知識を持つ

## 演習問題

### Easy（基礎確認）

Q1: フェルミエネルギーの定義として最も適切なものはどれですか？

a) 金属中の電子の平均エネルギー  
b) 絶対零度で最もエネルギーの高い電子が持つエネルギー  
c) 伝導帯の最低エネルギー  
d) バンドギャップの中央のエネルギー

解答を見る

**正解: b) 絶対零度で最もエネルギーの高い電子が持つエネルギー**

**解説:**  
フェルミエネルギー E_F は、T=0K で電子が最低エネルギー状態から順に詰まっていったとき、最も高いエネルギーを持つ電子のエネルギーです。フェルミ粒子（電子）はパウリの排他原理に従うため、1つの状態に最大2個（スピン↑↓）しか入れません。

**補足:**  
選択肢c)とd)は半導体や絶縁体の説明であり、金属の定義ではありません。a)の平均エネルギーは E_F よりも低い値になります（T=0Kで約3E_F/5）。

Q2: 銅（Cu）のフェルミエネルギーは約7.0 eVです。フェルミ温度 T_F は何Kですか？（k_B = 8.617 × 10⁻⁵ eV/K）

解答を見る

**正解: 約81,000 K**

**計算:**  
T_F = E_F / k_B = 7.0 eV / (8.617 × 10⁻⁵ eV/K) ≈ 81,200 K

**重要ポイント:**  
室温（300 K）は T_F のわずか0.4%にすぎません。これが、電子の比熱が古典論の予測（3/2 Nk_B）よりはるかに小さい理由です。実際に熱的に励起される電子は、フェルミエネルギー近傍のごく一部（~kT の範囲内）だけです。

Q3: 金属と絶縁体のバンド構造の違いを説明してください。

解答を見る

**正解例:**

  * **金属:** フェルミエネルギー E_F がバンドの途中に位置し、部分的に占有されたバンドがある。または複数のバンドが重なっている。D(E_F) > 0 であり、電場で容易にキャリアが移動できるため導電性がある。
  * **絶縁体:** 価電子帯が完全に占有され、伝導帯は完全に空。E_F はバンドギャップ（禁制帯）の中に位置する。バンドギャップが大きい（>3 eV）ため、室温では電子を伝導帯に励起できず、導電性がない。

**キーポイント:**  
金属か絶縁体かは、バンドの形ではなく「バンドがどこまで電子で埋まっているか」で決まります。同じバンド構造でも電子数が異なれば性質が変わります。

### Medium（応用）

Q4: シリコン（Si）のバンドギャップは 1.12 eV です。可視光（波長 400-700 nm）を吸収できますか？計算して説明してください。（h = 4.136 × 10⁻¹⁵ eV·s, c = 3 × 10⁸ m/s）

解答を見る

**結論: 可視光の一部（青〜紫）は吸収できるが、赤色は吸収できない**

**計算:**

光子エネルギー E = hc/λ より：

  * λ = 400 nm（紫）: E = (4.136 × 10⁻¹⁵ eV·s × 3 × 10⁸ m/s) / (400 × 10⁻⁹ m) = 3.10 eV
  * λ = 550 nm（緑）: E = 2.25 eV
  * λ = 700 nm（赤）: E = 1.77 eV

E_g = 1.12 eV なので：

  * 紫〜緑（E > 2 eV）: 吸収する（E > E_g）
  * 赤（E = 1.77 eV）: 一部吸収
  * 近赤外（E < 1.12 eV）: 透過（吸収しない）

**実用上の意味:**  
これがSiが太陽電池材料として使われる理由の一つです。太陽光の主要部分（可視〜近赤外）をカバーしています。ただし、赤外線は透過するため、バンドギャップがより小さいGe（0.66 eV）やGaAs（1.42 eV）と組み合わせたタンデム型太陽電池も研究されています。

Q5: 3次元自由電子モデルで、状態密度 D(E) ∝ √E です。これがフェルミエネルギーでの電子比熱 C_v ∝ T に繋がることを定性的に説明してください。

解答を見る

**説明:**

  1. **励起される電子の数** : 温度 T で熱的に励起されるのは、E_F の周辺 ~k_BT のエネルギー範囲内の電子のみ。その数は D(E_F) × k_BT に比例。
  2. **各電子が得るエネルギー** : 平均で ~k_BT。
  3. **全エネルギー** : U ~ [D(E_F) × k_BT] × k_BT = D(E_F) × (k_BT)²
  4. **比熱** : C_v = dU/dT ~ D(E_F) × k_B² × T ∝ T

**重要な点:**

  * 古典論では全電子（N個）が励起されるため C_v ~ Nk_B（温度に依存しない）
  * 量子論では励起される電子は N × (T/T_F) のみのため、C_v は T に比例し、係数が (T/T_F) 倍小さい
  * 室温では T/T_F ~ 0.004 なので、電子比熱は格子振動の比熱（~3Nk_B）よりはるかに小さく無視できる

Q6: Example 1のコードを修正して、アルカリ金属（Li, Na, K, Rb, Cs）の電子密度からフェルミエネルギーとフェルミ温度を計算し、プロットしてください。原子番号が増えると E_F はどう変化するか考察せよ。

解答を見る

**コード例:**
    
    
    alkali_metals = {
        'Li': 4.70e28,
        'Na': 2.65e28,
        'K': 1.40e28,
        'Rb': 1.15e28,
        'Cs': 0.91e28
    }
    
    names = list(alkali_metals.keys())
    E_F_values = [fermi_energy(n) for n in alkali_metals.values()]
    T_F_values = [E_F * e / constants.k / 1000 for E_F in E_F_values]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.bar(names, E_F_values, color='skyblue', edgecolor='navy', linewidth=2)
    ax1.set_ylabel('Fermi Energy (eV)', fontsize=12)
    ax1.set_title('Fermi Energy of Alkali Metals', fontsize=13, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    
    ax2.bar(names, T_F_values, color='salmon', edgecolor='darkred', linewidth=2)
    ax2.set_ylabel('Fermi Temperature (10³ K)', fontsize=12)
    ax2.set_title('Fermi Temperature of Alkali Metals', fontsize=13, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    

**考察:**

  * E_F ∝ n^(2/3) なので、電子密度が低いほど E_F は小さくなる
  * 原子番号が増える（Li → Cs）と原子が大きくなり、電子密度 n が減少
  * したがって Li（最小原子）が最も高い E_F を持つ（4.74 eV）
  * Cs（最大原子）は最も低い E_F（~1.59 eV）
  * これは軟らかさ（融点の低さ）とも相関します：CsはE_Fが低く、結合が弱く、融点28.5℃

### Hard（発展）

Q7: 2次元材料（グラフェンなど）では、自由電子モデルの状態密度はどうなるか導出してください。3次元の D(E) ∝ √E との違いを議論せよ。

解答を見る

**導出:**

2次元の場合、波数空間は平面（k_x, k_y）。自由電子のエネルギー:

E = ℏ²k²/(2m) = ℏ²(k_x² + k_y²)/(2m) 

k空間での状態密度（単位面積あたり）: 1/(2π)² = 1/(4π²)

エネルギー E を持つ状態は半径 k = √(2mE/ℏ²) の円周上にある。

エネルギー幅 dE に対応する k 幅: dk = (m/ℏ²k) dE

状態数: dN = [円周長 2πk] × dk × [1/(4π²)] × 2（スピン）

D(E) = dN/dE = (2πk × dk/dE) / (2π²) = (2m)/(2πℏ²) = 定数 

**重要な違い:**

次元 | D(E) | 特徴  
---|---|---  
1D | ∝ E^(-1/2) | 低エネルギーで発散  
2D | = 定数 | エネルギーに依存しない  
3D | ∝ E^(1/2) | 高エネルギーで増加  
  
**物理的意味:**

  * 2D系では電子比熱 C_v が3D系と異なる温度依存性を示す
  * グラフェンは自由電子ではなくディラック分散（E ∝ k）を持つため、実際は D(E) ∝ |E|
  * 2D材料特有の量子ホール効果、超伝導などが観測される

Q8: ダイヤモンド（C）とグラファイト（C）は同じ元素からなるのに、片方は絶縁体でもう片方は半金属です。バンド理論の観点から、この違いを生む構造的要因を説明してください。

解答を見る

**構造の違い:**

  * **ダイヤモンド:** sp³混成、3次元ネットワーク、炭素-炭素結合距離 1.54 Å
  * **グラファイト:** sp²混成、2次元層状構造、層内結合 1.42 Å、層間距離 3.35 Å

**電子構造の違い:**

**ダイヤモンド:**

  * 4個の価電子が全てsp³混成軌道でσ結合を形成
  * 結合性軌道（価電子帯）が完全に占有
  * 反結合性軌道（伝導帯）は完全に空
  * バンドギャップ E_g = 5.5 eV（大きい）
  * → 絶縁体

**グラファイト:**

  * 3個の価電子がsp²混成でσ結合、残り1個がπ軌道
  * π軌道が連続したバンドを形成（π電子が非局在化）
  * π軌道とπ*軌道（反結合）が K点（ブリルアンゾーン角）でちょうど接触
  * バンドギャップ E_g = 0 eV（ゼロギャップ）
  * → 半金属（状態密度がゼロに近いが有限の伝導性）

**重要なポイント:**

  1. **混成の違い** （sp³ vs sp²）が決定的
  2. **次元性** : グラファイトの2D層構造がπ電子の非局在化を促進
  3. **層間相互作用** : グラファイトの層を1枚にすると「グラフェン」となり、ディラックコーン型の特異なバンド構造を持つ

**実用的意味:**

  * ダイヤモンド: 絶縁体として高電圧素子、光学窓材
  * グラファイト: 電極材料、潤滑剤、リチウムイオン電池負極
  * グラフェン: 超高速トランジスタ、透明電極（研究段階）

Q9: （プログラミング課題）tight-binding モデルで、1次元鎖の最近接ホッピング積分 t を変化させたとき、バンド幅とバンドギャップがどう変化するかを計算し、プロットしてください。物理的意味を考察せよ。

解答を見る

**コード例:**
    
    
    def tight_binding_1d(k, a, t, epsilon_0=0):
        """
        1次元tight-bindingモデルのバンド構造
    
        Args:
            k (array): 波数
            a (float): 格子定数
            t (float): ホッピング積分
            epsilon_0 (float): オンサイトエネルギー
    
        Returns:
            array: エネルギー
        """
        return epsilon_0 - 2 * t * np.cos(k * a)
    
    a = 3e-10  # 3 Å
    k = np.linspace(-np.pi/a, np.pi/a, 500)
    t_values = [0.5*e, 1.0*e, 2.0*e, 4.0*e]  # eV
    
    plt.figure(figsize=(10, 7))
    
    for t in t_values:
        E = tight_binding_1d(k, a, t)
        bandwidth = np.max(E) - np.min(E)
        plt.plot(k*a/np.pi, E/e, linewidth=2.5, label=f't = {t/e:.1f} eV (BW = {bandwidth/e:.2f} eV)')
    
    plt.xlabel('Wave vector k (π/a)', fontsize=12)
    plt.ylabel('Energy (eV)', fontsize=12)
    plt.title('1D Tight-Binding Model: Effect of Hopping Parameter', fontsize=14, fontweight='bold')
    plt.axhline(0, color='k', linewidth=0.5, linestyle='--')
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # バンド幅の t 依存性
    t_range = np.linspace(0.1*e, 5*e, 50)
    bandwidth = [4*t for t in t_range]  # BW = 4t (理論値)
    
    plt.figure(figsize=(8, 6))
    plt.plot(t_range/e, np.array(bandwidth)/e, 'b-', linewidth=3)
    plt.xlabel('Hopping parameter t (eV)', fontsize=12)
    plt.ylabel('Bandwidth (eV)', fontsize=12)
    plt.title('Bandwidth vs Hopping Parameter', fontsize=14, fontweight='bold')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    print("📊 解析結果:")
    print(f"バンド幅 = 4t （t が大きいほどバンドが広がる）")
    print(f"物理的意味: t が大きい → 電子の波動関数が隣接サイトと強く重なる")
    print(f"            → 電子が結晶中を移動しやすい（有効質量 m* ∝ 1/t）")
    

**考察:**

  * **t（ホッピング積分）の物理的意味** : 電子が隣接原子へ「トンネル」する確率の大きさ
  * **バンド幅 ∝ t** : t が大きい → 電子が動きやすい → バンドが広い → 有効質量が軽い
  * **2原子分子の類推** : 原子間距離が近い（重なり大）→ 結合性-反結合性のエネルギー差が大きい
  * **実際の材料** : 遷移金属（d軌道の広がりが大きい）は t が大きく広いバンド → 高い導電性

Q10: （発展問題）Mott絶縁体は、バンド理論では金属と予測されるが実際は絶縁体です（例: NiO, La₂CuO₄）。この現象をバンド理論の枠組みだけでは説明できない理由を調べ、電子相関の役割について簡潔にまとめてください。

解答を見る

**Mott絶縁体とは:**

  * バンド理論: 部分的に占有されたd軌道バンド → 金属と予測
  * 実験: 絶縁体（ギャップ ~1-3 eV）
  * 例: NiO（Ni²⁺の3d⁸配置）、La₂CuO₄（高温超伝導の母物質）

**バンド理論が失敗する理由:**

  1. **強い電子相関を無視**
     * バンド理論は各電子が独立に動くと仮定（一電子近似）
     * 実際は電子間のクーロン反発 U が運動エネルギー（バンド幅 W）より大きい場合がある
  2. **Mott-Hubbard描像**
     * 1サイトに2個の電子が入るコスト（U）が大きすぎる
     * 電子が局在化し、各サイトに1個ずつ配置
     * 電子を動かすには U のエネルギーが必要 → ギャップ形成

**相図（U vs W）:**

  * **U << W**: 運動エネルギー優勢 → バンド理論有効 → 金属
  * **U >> W**: クーロン相互作用優勢 → 電子局在 → Mott絶縁体
  * **U ~ W** : 競合領域 → 強相関系（高温超伝導、重い電子系など）

**動的平均場理論（DMFT）の成功:**

  * 1990年代に開発された理論（Georges, Kotliar et al.）
  * 電子相関を取り入れつつバンド構造も保持
  * Mott転移（金属-絶縁体転移）を記述可能
  * バンド理論 + 相関効果の統合的描像を提供

**実用的重要性:**

  * **高温超伝導** : Mott絶縁体にキャリアをドープすると超伝導が出現（銅酸化物、鉄系超伝導体）
  * **巨大磁気抵抗** : マンガン酸化物（CMR効果）
  * **Mottronics** : Mott転移を利用したスイッチング素子の提案

**結論:**

バンド理論は多くの材料を理解する強力な枠組みですが、電子相関が強い系（遷移金属酸化物、f電子系など）では不十分です。これらを理解するには、多体効果を考慮した手法（DMFT、量子モンテカルロ、密度汎関数理論+U など）が必要です。

## 次のステップ

第1章では固体電子論の基礎として、バンド理論の基本概念を学びました。次章では、この知識を基に**実際の材料の電子構造を計算する手法** を学びます。

**第2章の予告: 密度汎関数理論（DFT）入門**

  * Hohenberg-KohnとKohn-Shamの定理
  * 交換相関汎関数（LDA, GGA, ハイブリッド）
  * Pythonでの実践: ASE + GPAW によるバンド構造計算
  * 実材料への応用: Si, GaAs, TiO₂のバンド構造

## 参考文献

### 基礎的テキスト

  1. Ashcroft, N. W., & Mermin, N. D. (1976). _Solid State Physics_. Holt, Rinehart and Winston. （固体物理学の古典的名著）
  2. Kittel, C. (2004). _Introduction to Solid State Physics_ (8th ed.). Wiley. （入門書として最適）
  3. Marder, M. P. (2010). _Condensed Matter Physics_ (2nd ed.). Wiley. （現代的なアプローチ）

### バンド理論の詳細

  4. Harrison, W. A. (1980). _Electronic Structure and the Properties of Solids_. Freeman. （tight-binding法の詳細）
  5. Yu, P. Y., & Cardona, M. (2010). _Fundamentals of Semiconductors_ (4th ed.). Springer. （半導体物理の標準的教科書）

### 計算物理化学

  6. Martin, R. M. (2004). _Electronic Structure: Basic Theory and Practical Methods_. Cambridge University Press. （DFTの理論と実践）
  7. Sholl, D., & Steckel, J. A. (2009). _Density Functional Theory: A Practical Introduction_. Wiley. （DFT入門）

### オンラインリソース

  8. Materials Project. (2024). Electronic Structure Calculations. <https://materialsproject.org>
  9. ASE (Atomic Simulation Environment). <https://wiki.fysik.dtu.dk/ase/>
  10. GPAW Documentation. <https://wiki.fysik.dtu.dk/gpaw/>

### 原著論文（歴史的重要論文）

  11. Bloch, F. (1929). "Über die Quantenmechanik der Elektronen in Kristallgittern." _Zeitschrift für Physik_ , 52(7-8), 555-600. （Blochの定理）
  12. Wilson, A. H. (1931). "The Theory of Electronic Semi-Conductors." _Proceedings of the Royal Society A_ , 133(822), 458-491. （半導体理論の基礎）

[← シリーズ目次](<./index.html>) [第2章: DFT入門 →](<./chapter-2.html>)

### 免責事項

  * 本コンテンツは教育・研究・情報提供のみを目的としており、専門的な助言(法律・会計・技術的保証など)を提供するものではありません。
  * 本コンテンツおよび付随するCode examplesは「現状有姿(AS IS)」で提供され、明示または黙示を問わず、商品性、特定目的適合性、権利非侵害、正確性・完全性、動作・安全性等いかなる保証もしません。
  * 外部リンク、第三者が提供するデータ・ツール・ライブラリ等の内容・可用性・安全性について、作成者および東北大学は一切の責任を負いません。
  * 本コンテンツの利用・実行・解釈により直接的・間接的・付随的・特別・結果的・懲罰的損害が生じた場合でも、適用法で許容される最大限の範囲で、作成者および東北大学は責任を負いません。
  * 本コンテンツの内容は、予告なく変更・更新・提供停止されることがあります。
  * 本コンテンツの著作権・ライセンスは明記された条件(例: CC BY 4.0)に従います。当該ライセンスは通常、無保証条項を含みます。
