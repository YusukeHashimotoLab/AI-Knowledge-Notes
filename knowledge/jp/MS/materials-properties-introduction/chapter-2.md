---
title: 第2章：結晶場理論と電子状態
chapter_title: 第2章：結晶場理論と電子状態
subtitle: 遷移金属化合物のd軌道分裂と配位子場効果
reading_time: 25-30分
difficulty: 中級
code_examples: 8
---

なぜ遷移金属化合物は多彩な色を示すのか？結晶場理論を学び、d軌道の分裂と電子状態の関係を理解します。Jahn-Teller効果、配位子場理論、そして実際の材料への応用まで、Pythonでエネルギー準位を計算しながら学びましょう。 

## 学習目標

この章を読むことで、以下を習得できます：

### 基本レベル（初学者）

  * ✅ 結晶場理論の基本概念と物理的背景を説明できる
  * ✅ 八面体・四面体配位でのd軌道分裂パターンを理解する
  * ✅ 結晶場分裂エネルギー（Δ）の意味を説明できる

### 中級レベル（実践者）

  * ✅ Jahn-Teller効果のメカニズムと応用例を理解する
  * ✅ 配位子場理論と結晶場理論の違いを説明できる
  * ✅ Pythonでd軌道エネルギー準位を計算・可視化できる
  * ✅ 遷移金属化合物の色と電子状態の関係を解析できる

### 応用レベル（研究者）

  * ✅ Tanabe-Sugano図を読み取り、d電子配置を予測できる
  * ✅ 実験データから結晶場パラメータを決定できる
  * ✅ 遷移金属化合物の磁性・光学特性を第一原理計算で予測できる

* * *

## 2.1 結晶場理論の基礎

### 結晶場理論とは

**結晶場理論（Crystal Field Theory: CFT）** は、遷移金属イオンが配位子に囲まれたときに、d軌道のエネルギー準位がどのように分裂するかを説明する理論です。1929年にHans BetheとJohn Van Vleckによって提唱されました。

#### 📖 定義：結晶場理論

配位子を**負の点電荷** として扱い、その静電場（結晶場）がd軌道の縮退を解くことで、エネルギー準位が分裂する現象を記述する静電的モデル。

**基本的な考え方** ：

  1. 孤立した遷移金属イオンでは、5つのd軌道は縮退している（同じエネルギー）
  2. 配位子が近づくと、静電反発によりd軌道のエネルギーが上昇する
  3. 配位子の配置（対称性）により、d軌道は異なるエネルギー準位に分裂する

### d軌道の空間分布

d軌道は5種類あり、それぞれ異なる空間分布を持ちます：

軌道 | 空間分布 | 特徴  
---|---|---  
$d_{z^2}$ | z軸に沿って伸びる | 軸配位子と強く反発  
$d_{x^2-y^2}$ | x-y平面に沿って伸びる | 平面配位子と強く反発  
$d_{xy}$ | xy平面の対角線方向 | 配位子との反発が弱い  
$d_{xz}$ | xz平面の対角線方向 | 配位子との反発が弱い  
$d_{yz}$ | yz平面の対角線方向 | 配位子との反発が弱い  
  
**💡 重要なポイント**  
$d_{z^2}$と$d_{x^2-y^2}$は配位子方向を向くため「eg軌道」と呼ばれ、$d_{xy}, d_{xz}, d_{yz}$は配位子間を向くため「t2g軌道」と呼ばれます（八面体配位の場合）。 

## 2.2 八面体配位における結晶場分裂

### 八面体配位の対称性

遷移金属イオンが6つの配位子に囲まれた**八面体配位（O h対称）**では、d軌道は以下のように分裂します：
    
    
    ```mermaid
    graph TD
        A[孤立イオン5つのd軌道は縮退] --> B[球対称場全体的にエネルギー上昇]
        B --> C[八面体場e_g と t_2g に分裂]
    
        D["e_g (d_z², d_x²-y²)エネルギー高い"] -.Δ_oct.-> E["t_2g (d_xy, d_xz, d_yz)エネルギー低い"]
    
        C --> D
        C --> E
    
        style A fill:#f093fb,stroke:#f5576c,stroke-width:2px,color:#fff
        style B fill:#f5a3fb,stroke:#f5576c,stroke-width:2px,color:#fff
        style C fill:#f5b3fb,stroke:#f5576c,stroke-width:2px,color:#fff
        style D fill:#ff6b6b,stroke:#c92a2a,stroke-width:2px,color:#fff
        style E fill:#51cf66,stroke:#2f9e44,stroke-width:2px,color:#fff
    ```

**結晶場分裂エネルギー（Δ oct または 10Dq）**：

#### 📖 定義：Δoct

八面体配位における eg と t2g 軌道のエネルギー差。

$$\Delta_{\text{oct}} = E(e_g) - E(t_{2g})$$

**エネルギー安定化** ：

  * eg軌道：+ 0.6 Δoct（不安定化）
  * t2g軌道：- 0.4 Δoct（安定化）
  * エネルギーの重心は保存される

### 八面体配位のエネルギー図を描く
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    def plot_octahedral_splitting(delta_oct=10):
        """
        八面体配位におけるd軌道の結晶場分裂を可視化
    
        Parameters:
        -----------
        delta_oct : float
            結晶場分裂エネルギー (Dq単位)
        """
        fig, ax = plt.subplots(figsize=(10, 6))
    
        # エネルギー準位
        E_barycenter = 0  # エネルギーの重心
        E_eg = E_barycenter + 0.6 * delta_oct
        E_t2g = E_barycenter - 0.4 * delta_oct
    
        # 孤立イオン（左）
        ax.hlines(0, 0, 1.5, colors='gray', linewidth=2, label='孤立イオン（5d縮退）')
        ax.text(0.75, 0.5, '5d', ha='center', fontsize=12, fontweight='bold')
    
        # 球対称場（中央）
        ax.hlines(E_barycenter, 2.5, 4, colors='blue', linewidth=2, label='球対称場')
        ax.text(3.25, E_barycenter+0.5, '5d', ha='center', fontsize=12, fontweight='bold')
    
        # 八面体場（右）
        ax.hlines(E_eg, 5, 7, colors='red', linewidth=3, label='e$_g$ (2軌道)')
        ax.text(6, E_eg+0.5, '$d_{z^2}$, $d_{x^2-y^2}$', ha='center', fontsize=11)
    
        ax.hlines(E_t2g, 5, 7, colors='green', linewidth=3, label='t$_{2g}$ (3軌道)')
        ax.text(6, E_t2g-0.8, '$d_{xy}$, $d_{xz}$, $d_{yz}$', ha='center', fontsize=11)
    
        # Δ_octを示す矢印
        ax.annotate('', xy=(7.5, E_eg), xytext=(7.5, E_t2g),
                    arrowprops=dict(arrowstyle='<->', color='purple', lw=2))
        ax.text(8, (E_eg + E_t2g)/2, f'$\\Delta_{{oct}}$ = {delta_oct} Dq',
                fontsize=13, fontweight='bold', color='purple')
    
        # 軸設定
        ax.set_xlim(-0.5, 9)
        ax.set_ylim(-5, 8)
        ax.set_ylabel('エネルギー (Dq)', fontsize=12)
        ax.set_xticks([0.75, 3.25, 6])
        ax.set_xticklabels(['孤立イオン', '球対称場', '八面体場'], fontsize=11)
        ax.grid(axis='y', alpha=0.3)
        ax.set_title('八面体配位における結晶場分裂', fontsize=14, fontweight='bold')
    
        plt.tight_layout()
        plt.show()
    
    # 実行
    plot_octahedral_splitting(delta_oct=10)
    

#### 🔬 実例：[Ti(H2O)6]3+の色

**電子配置** ：Ti3+はd1電子配置（1個のd電子）

  * 基底状態：t2g軌道に1電子
  * 励起状態：eg軌道に1電子
  * 可視光吸収：緑～黄色（約20,000 cm-1）を吸収
  * 観測される色：**紫色** （補色）

## 2.3 四面体配位における結晶場分裂

### 四面体配位の特徴

**四面体配位（T d対称）**では、配位子が4つであり、八面体とは逆のパターンで分裂します：

  * e軌道（$d_{z^2}, d_{x^2-y^2}$）：エネルギーが**低い**
  * t2軌道（$d_{xy}, d_{xz}, d_{yz}$）：エネルギーが**高い**
  * 分裂幅：$\Delta_{\text{tet}} \approx \frac{4}{9} \Delta_{\text{oct}}$（約44%）

**💡 なぜ逆転するのか？**  
四面体では配位子が立体対角線上に配置されるため、t2軌道（対角線方向に電子密度）の方が配位子と強く反発し、エネルギーが高くなります。 

### 八面体と四面体の比較
    
    
    import matplotlib.pyplot as plt
    import numpy as np
    
    def compare_octahedral_tetrahedral():
        """八面体と四面体の結晶場分裂を比較"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
        # 八面体配位
        delta_oct = 10
        E_eg_oct = 0.6 * delta_oct
        E_t2g_oct = -0.4 * delta_oct
    
        ax1.hlines(E_eg_oct, 0, 1, colors='red', linewidth=4, label='e$_g$')
        ax1.text(0.5, E_eg_oct + 0.5, 'e$_g$ (2)', ha='center', fontsize=12, fontweight='bold')
    
        ax1.hlines(E_t2g_oct, 0, 1, colors='green', linewidth=4, label='t$_{2g}$')
        ax1.text(0.5, E_t2g_oct - 0.7, 't$_{2g}$ (3)', ha='center', fontsize=12, fontweight='bold')
    
        ax1.annotate('', xy=(1.3, E_eg_oct), xytext=(1.3, E_t2g_oct),
                    arrowprops=dict(arrowstyle='<->', color='purple', lw=2))
        ax1.text(1.6, (E_eg_oct + E_t2g_oct)/2, '$\\Delta_{oct}$', fontsize=13, fontweight='bold')
    
        ax1.set_xlim(-0.2, 2)
        ax1.set_ylim(-6, 8)
        ax1.set_ylabel('エネルギー (Dq)', fontsize=12)
        ax1.set_title('八面体配位 (O$_h$)', fontsize=14, fontweight='bold')
        ax1.grid(axis='y', alpha=0.3)
        ax1.set_xticks([])
    
        # 四面体配位
        delta_tet = (4/9) * delta_oct  # 約4.44 Dq
        E_t2_tet = 0.6 * delta_tet
        E_e_tet = -0.4 * delta_tet
    
        ax2.hlines(E_t2_tet, 0, 1, colors='green', linewidth=4, label='t$_2$')
        ax2.text(0.5, E_t2_tet + 0.5, 't$_2$ (3)', ha='center', fontsize=12, fontweight='bold')
    
        ax2.hlines(E_e_tet, 0, 1, colors='red', linewidth=4, label='e')
        ax2.text(0.5, E_e_tet - 0.7, 'e (2)', ha='center', fontsize=12, fontweight='bold')
    
        ax2.annotate('', xy=(1.3, E_t2_tet), xytext=(1.3, E_e_tet),
                    arrowprops=dict(arrowstyle='<->', color='orange', lw=2))
        ax2.text(1.7, (E_t2_tet + E_e_tet)/2, f'$\\Delta_{{tet}}$\n≈ {delta_tet:.1f} Dq',
                fontsize=12, fontweight='bold')
    
        ax2.set_xlim(-0.2, 2.2)
        ax2.set_ylim(-6, 8)
        ax2.set_ylabel('エネルギー (Dq)', fontsize=12)
        ax2.set_title('四面体配位 (T$_d$)', fontsize=14, fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)
        ax2.set_xticks([])
    
        plt.tight_layout()
        plt.show()
    
    compare_octahedral_tetrahedral()
    

## 2.4 Jahn-Teller効果

### Jahn-Teller定理

#### 📖 Jahn-Teller定理（1937）

縮退した電子状態を持つ非線形分子は、**必ず構造歪みを起こして縮退を解消する** 。これにより全エネルギーが低下する。

**発現条件** ：

  * eg軌道またはt2g軌道が部分的に占有されている
  * 典型例：Cu2+（d9）、Mn3+（d4高スピン）、Cr2+（d4高スピン）

### Cu2+の Jahn-Teller歪み

Cu2+はd9電子配置を持ち、eg軌道に3電子（$d_{x^2-y^2}$: 2個、$d_{z^2}$: 1個）が入ります。

#### 🔬 実例：CuF2の構造歪み

**理想八面体** ：6つのF-が等距離（例：2.0 Å）

**Jahn-Teller歪み後** ：

  * z軸方向：Cu-F結合が伸びる（2.27 Å）→ $d_{z^2}$のエネルギー低下
  * xy平面：Cu-F結合が縮む（1.93 Å）→ $d_{x^2-y^2}$のエネルギー上昇
  * 結果：eg縮退が解ける → エネルギー安定化

### Jahn-Teller効果のシミュレーション
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    def jahn_teller_distortion_energy():
        """Jahn-Teller歪みによるエネルギー変化を計算"""
        # 歪みパラメータ（結合長の変化率）
        distortion = np.linspace(-0.15, 0.15, 100)  # -15% ~ +15%
    
        # Cu2+の場合：d9配置、eg軌道に3電子
        # 理想八面体でのe_g縮退エネルギー
        E_ideal = 0
    
        # 歪み後のエネルギー（簡略化モデル）
        # 軸方向伸長（正の歪み）の場合
        E_dz2 = E_ideal - 1000 * distortion**2  # d_z2軌道が安定化
        E_dx2y2 = E_ideal + 800 * distortion**2  # d_x2-y2軌道が不安定化
    
        # 電子配置：d_z2に2電子、d_x2-y2に1電子
        E_total = 2 * E_dz2 + 1 * E_dx2y2
    
        # 弾性エネルギー（構造歪みのコスト）
        E_elastic = 500 * distortion**2
    
        # 全エネルギー
        E_net = E_total + E_elastic
    
        # プロット
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
        # 軌道エネルギー
        ax1.plot(distortion * 100, E_dz2, 'b-', linewidth=2, label='$d_{z^2}$ (2電子)')
        ax1.plot(distortion * 100, E_dx2y2, 'r-', linewidth=2, label='$d_{x^2-y^2}$ (1電子)')
        ax1.axhline(0, color='gray', linestyle='--', alpha=0.5)
        ax1.axvline(0, color='gray', linestyle='--', alpha=0.5)
        ax1.set_xlabel('軸方向の歪み (%)', fontsize=12)
        ax1.set_ylabel('軌道エネルギー (cm$^{-1}$)', fontsize=12)
        ax1.set_title('e$_g$軌道の分裂', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=11)
        ax1.grid(alpha=0.3)
    
        # 全エネルギー
        ax2.plot(distortion * 100, E_total, 'g-', linewidth=2, label='電子エネルギー')
        ax2.plot(distortion * 100, E_elastic, 'orange', linewidth=2, label='弾性エネルギー')
        ax2.plot(distortion * 100, E_net, 'purple', linewidth=3, label='全エネルギー')
    
        # 最安定構造
        min_idx = np.argmin(E_net)
        min_distortion = distortion[min_idx]
        ax2.plot(min_distortion * 100, E_net[min_idx], 'ro', markersize=10, label='最安定構造')
    
        ax2.axhline(0, color='gray', linestyle='--', alpha=0.5)
        ax2.axvline(0, color='gray', linestyle='--', alpha=0.5)
        ax2.set_xlabel('軸方向の歪み (%)', fontsize=12)
        ax2.set_ylabel('エネルギー (cm$^{-1}$)', fontsize=12)
        ax2.set_title('Jahn-Teller安定化エネルギー', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(alpha=0.3)
    
        plt.tight_layout()
        plt.show()
    
        print(f"最安定歪み: {min_distortion*100:.2f}%")
        print(f"安定化エネルギー: {-E_net[min_idx]:.1f} cm^-1")
    
    jahn_teller_distortion_energy()
    

## 2.5 配位子場理論

### 結晶場理論の限界と配位子場理論

結晶場理論は配位子を単なる**点電荷** として扱いますが、実際には配位子と金属イオンの間で**共有結合性** が存在します。

#### 📖 配位子場理論（Ligand Field Theory: LFT）

分子軌道論に基づき、金属d軌道と配位子軌道の**混成** を考慮した理論。結晶場理論より精密な記述が可能。

項目 | 結晶場理論（CFT） | 配位子場理論（LFT）  
---|---|---  
配位子の扱い | 点電荷 | 分子軌道として扱う  
軌道の混成 | 考慮しない | σ結合・π結合を考慮  
予測精度 | 定性的 | 半定量的～定量的  
適用範囲 | 弱配位子 | 強配位子、π供与/逆供与  
  
### 分光化学系列

配位子の**結晶場分裂の強さ** を並べたものを**分光化学系列** といいます：

I- < Br- < Cl- < F- < OH- < H2O < NH3 < en < CN- < CO 

← 弱配位子場 ────── 強配位子場 → 

  * **弱配位子** （I-, Br-, Cl-）：Δが小さい → 高スピン状態
  * **強配位子** （CN-, CO）：Δが大きい → 低スピン状態

### 高スピン・低スピン状態の計算
    
    
    import matplotlib.pyplot as plt
    import numpy as np
    
    def calculate_spin_states(d_electrons, delta_oct):
        """
        d電子数と結晶場分裂から高スピン/低スピンを計算
    
        Parameters:
        -----------
        d_electrons : int
            d電子の数（1～10）
        delta_oct : float
            結晶場分裂エネルギー（単位: cm^-1）
    
        Returns:
        --------
        dict : スピン状態の情報
        """
        # Pairing energy（電子対形成に必要なエネルギー）
        P = 15000  # 典型値: 約15,000 cm^-1
    
        # 高スピン配置（Hund則優先）
        if d_electrons <= 3:
            high_spin = d_electrons  # t2g軌道に順次配置
            hs_t2g = d_electrons
            hs_eg = 0
        elif d_electrons <= 5:
            high_spin = d_electrons  # t2g満たしてeg軌道へ
            hs_t2g = 3
            hs_eg = d_electrons - 3
        elif d_electrons <= 8:
            high_spin = d_electrons - 5  # t2g, eg満たして対形成開始
            hs_t2g = min(6, d_electrons)
            hs_eg = max(0, d_electrons - 6)
        else:
            high_spin = 10 - d_electrons
            hs_t2g = 6
            hs_eg = d_electrons - 6
    
        # 低スピン配置（Δ_oct > P のとき）
        if d_electrons <= 6:
            low_spin = d_electrons % 2  # t2g軌道を先に埋める
            ls_t2g = min(6, d_electrons)
            ls_eg = max(0, d_electrons - 6)
        else:
            low_spin = (d_electrons - 6) % 2
            ls_t2g = 6
            ls_eg = d_electrons - 6
    
        # Crystal Field Stabilization Energy (CFSE)
        cfse_high = (-0.4 * hs_t2g + 0.6 * hs_eg) * delta_oct
        cfse_low = (-0.4 * ls_t2g + 0.6 * ls_eg) * delta_oct
    
        # Pairing energyを考慮した全エネルギー
        n_pairs_high = (d_electrons - high_spin) / 2
        n_pairs_low = (d_electrons - low_spin) / 2
    
        E_high = cfse_high + n_pairs_high * P
        E_low = cfse_low + n_pairs_low * P
    
        return {
            'high_spin': high_spin,
            'low_spin': low_spin,
            'E_high': E_high,
            'E_low': E_low,
            'stable_state': 'Low Spin' if E_low < E_high else 'High Spin'
        }
    
    # Fe2+（d6）の例
    d_electrons = 6
    delta_values = np.linspace(5000, 25000, 100)
    
    high_spin_list = []
    low_spin_list = []
    
    for delta in delta_values:
        result = calculate_spin_states(d_electrons, delta)
        high_spin_list.append(result['E_high'])
        low_spin_list.append(result['E_low'])
    
    # プロット
    plt.figure(figsize=(10, 6))
    plt.plot(delta_values, high_spin_list, 'r-', linewidth=2, label='High Spin (S=2)')
    plt.plot(delta_values, low_spin_list, 'b-', linewidth=2, label='Low Spin (S=0)')
    
    # 交点（スピン転移点）
    idx_cross = np.argmin(np.abs(np.array(high_spin_list) - np.array(low_spin_list)))
    delta_cross = delta_values[idx_cross]
    plt.axvline(delta_cross, color='green', linestyle='--', linewidth=2, label=f'Spin Crossover: {delta_cross:.0f} cm$^{{-1}}$')
    
    plt.xlabel('結晶場分裂 Δ$_{oct}$ (cm$^{-1}$)', fontsize=12)
    plt.ylabel('全エネルギー (cm$^{-1}$)', fontsize=12)
    plt.title(f'Fe$^{{2+}}$ (d$^{6}$) の高スピン・低スピン状態', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    print(f"スピン転移点: Δ_oct ≈ {delta_cross:.0f} cm^-1")
    

## 2.6 Tanabe-Sugano図

### Tanabe-Sugano図とは

**Tanabe-Sugano図** は、dn電子配置を持つ遷移金属イオンの電子状態を、結晶場分裂の強さ（Δ/B）の関数として示した図です。

**💡 Tanabe-Sugano図の読み方**  

  * 横軸：Δ/B（結晶場分裂 / Racahパラメータ）
  * 縦軸：E/B（エネルギー準位 / Racahパラメータ）
  * 各曲線：異なる電子状態（項記号で表記）
  * 基底状態は太線で表示

### d3電子配置のTanabe-Sugano図の簡易版
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    def plot_tanabe_sugano_d3():
        """d3電子配置のTanabe-Sugano図（簡略版）"""
        # Δ/B の範囲
        delta_over_B = np.linspace(0, 3.5, 100)
    
        # 簡略化した電子状態のエネルギー（実際はより複雑）
        # 4F (基底状態)
        E_4F = 0 * delta_over_B
    
        # 4P
        E_4P = 15 + 0 * delta_over_B
    
        # 2G
        E_2G = 17 + 2 * delta_over_B
    
        # 2H
        E_2H = 22 + 1.5 * delta_over_B
    
        # 2D
        E_2D = 28 + 0.8 * delta_over_B
    
        # プロット
        fig, ax = plt.subplots(figsize=(10, 7))
    
        ax.plot(delta_over_B, E_4F, 'b-', linewidth=3, label='$^4$F (基底状態)')
        ax.plot(delta_over_B, E_4P, 'r-', linewidth=2, label='$^4$P')
        ax.plot(delta_over_B, E_2G, 'g-', linewidth=2, label='$^2$G')
        ax.plot(delta_over_B, E_2H, 'orange', linewidth=2, label='$^2$H')
        ax.plot(delta_over_B, E_2D, 'purple', linewidth=2, label='$^2$D')
    
        # 吸収遷移の例（Cr3+）
        delta_B_example = 2.3  # Cr3+の典型的な値
        ax.axvline(delta_B_example, color='gray', linestyle='--', alpha=0.5)
        ax.text(delta_B_example + 0.1, 35, 'Cr$^{3+}$\n(ruby)', fontsize=10, color='red')
    
        ax.set_xlabel('Δ / B', fontsize=13, fontweight='bold')
        ax.set_ylabel('E / B', fontsize=13, fontweight='bold')
        ax.set_title('Tanabe-Sugano図（d$^3$電子配置）', fontsize=15, fontweight='bold')
        ax.set_xlim(0, 3.5)
        ax.set_ylim(0, 40)
        ax.legend(fontsize=11, loc='upper left')
        ax.grid(alpha=0.3)
    
        plt.tight_layout()
        plt.show()
    
    plot_tanabe_sugano_d3()
    

#### 🔬 実例：ルビー（Cr3+:Al2O3）の色

Cr3+はd3電子配置を持ち、Al2O3中で八面体配位されています。

  * 基底状態：4A2g
  * 励起状態：4T2g（青緑吸収）、4T1g（黄色吸収）
  * 結果：青と黄色を吸収 → **赤色** を透過
  * 蛍光：2Eg → 4A2g 遷移（鋭い赤色発光）

## 2.7 遷移金属化合物の応用

### 触媒としての応用

遷移金属化合物は、d軌道が部分的に占有されているため、**酸化還元反応** や**配位子交換反応** の触媒として広く用いられます。

触媒 | 反応 | 結晶場の役割  
---|---|---  
Fe2+/Fe3+ | Fenton反応（H2O2分解） | eg軌道の電子移動  
Ti3+/Ti4+ | Ziegler-Natta重合 | 配位子の活性化  
V2+/V3+ | レドックスフロー電池 | 可逆的な酸化還元  
Ru2+/Ru3+ | 水の酸化反応 | t2g-π*逆供与  
  
### 磁性材料としての応用

遷移金属化合物の不対電子は、**磁性** の起源です。

  * **強磁性体** ：Fe, Co, Ni（金属）、CrO2（磁気テープ）
  * **反強磁性体** ：MnO, NiO, FeO
  * **フェリ磁性体** ：Fe3O4（マグネタイト）、γ-Fe2O3（マグヘマイト）

### 光学材料としての応用

結晶場による光吸収を利用した材料：

  * **宝石** ：ルビー（Cr3+）、エメラルド（Cr3+）、サファイア（Fe2+/Ti4+）
  * **顔料** ：プルシアンブルー（Fe2+/Fe3+）、酸化クロム緑（Cr3+）
  * **太陽電池** ：色素増感太陽電池（Ru錯体）

### 結晶場分裂と色の関係を可視化
    
    
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import FancyBboxPatch
    
    def crystal_field_color_chart():
        """遷移金属イオンの結晶場分裂と色の関係"""
    
        ions = ['Ti³⁺', 'V³⁺', 'Cr³⁺', 'Mn²⁺', 'Fe³⁺', 'Co²⁺', 'Ni²⁺', 'Cu²⁺']
        d_electrons = [1, 2, 3, 5, 5, 7, 8, 9]
        delta_oct = [20300, 18900, 17400, 21000, 14000, 9300, 8500, 12600]  # cm^-1
        colors = ['purple', 'green', 'red', 'pale pink', 'yellow', 'pink', 'green', 'blue']
        hex_colors = ['#9b59b6', '#27ae60', '#e74c3c', '#f8b3c7', '#f1c40f', '#ff69b4', '#2ecc71', '#3498db']
    
        fig, ax = plt.subplots(figsize=(12, 7))
    
        x_pos = np.arange(len(ions))
        bars = ax.bar(x_pos, delta_oct, color=hex_colors, edgecolor='black', linewidth=1.5, alpha=0.8)
    
        # データラベル
        for i, (bar, ion, d, col) in enumerate(zip(bars, ions, d_electrons, colors)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 500,
                    f'{int(height)} cm⁻¹', ha='center', va='bottom', fontsize=9, fontweight='bold')
            ax.text(bar.get_x() + bar.get_width()/2., -2000,
                    f'd{d}', ha='center', va='top', fontsize=10, color='gray')
            ax.text(bar.get_x() + bar.get_width()/2., height/2,
                    col, ha='center', va='center', fontsize=9, fontweight='bold',
                    color='white', rotation=90)
    
        ax.set_xticks(x_pos)
        ax.set_xticklabels(ions, fontsize=12, fontweight='bold')
        ax.set_ylabel('結晶場分裂 Δ$_{oct}$ (cm$^{-1}$)', fontsize=13)
        ax.set_title('八面体配位における遷移金属イオンの結晶場分裂と観測される色', fontsize=14, fontweight='bold')
        ax.set_ylim(-3000, 24000)
        ax.grid(axis='y', alpha=0.3)
    
        # 分光化学系列の参照線
        ax.axhline(15000, color='red', linestyle='--', linewidth=1.5, alpha=0.5, label='典型的な配位子（H₂O, NH₃）')
    
        ax.legend(fontsize=10)
        plt.tight_layout()
        plt.show()
    
    crystal_field_color_chart()
    

## 2.8 第一原理計算での結晶場計算

### DFTによる結晶場パラメータの計算

現代の材料科学では、**第一原理計算（DFT）** を用いて結晶場分裂エネルギーを直接計算できます。

**💡 DFT計算のワークフロー**  

  1. 結晶構造の最適化
  2. 電子状態計算（バンド構造、DOS）
  3. d軌道の状態密度（PDOS）を解析
  4. Δoctをt2gとegのエネルギー差から決定

### ASEを使った遷移金属化合物の構造生成
    
    
    from ase import Atoms
    from ase.visualize import view
    import numpy as np
    
    def create_octahedral_complex(metal='Ti', ligand='O', bond_length=2.0):
        """
        八面体配位の遷移金属錯体を生成
    
        Parameters:
        -----------
        metal : str
            中心金属イオン
        ligand : str
            配位子原子
        bond_length : float
            金属-配位子結合長（Å）
    
        Returns:
        --------
        atoms : ase.Atoms
            生成された構造
        """
        # 八面体配位の配位子位置（±x, ±y, ±z軸）
        ligand_positions = np.array([
            [bond_length, 0, 0],
            [-bond_length, 0, 0],
            [0, bond_length, 0],
            [0, -bond_length, 0],
            [0, 0, bond_length],
            [0, 0, -bond_length]
        ])
    
        # 原子オブジェクトの作成
        symbols = [metal] + [ligand] * 6
        positions = np.vstack([[0, 0, 0], ligand_positions])
    
        atoms = Atoms(symbols=symbols, positions=positions)
    
        # セルサイズを設定（可視化のため）
        cell_size = bond_length * 3
        atoms.set_cell([cell_size, cell_size, cell_size])
        atoms.center()
    
        return atoms
    
    # TiO6八面体を生成
    complex_TiO6 = create_octahedral_complex(metal='Ti', ligand='O', bond_length=2.0)
    
    print(f"生成された構造: {complex_TiO6.get_chemical_formula()}")
    print(f"原子位置:\n{complex_TiO6.get_positions()}")
    
    # 可視化（ASE GUIが利用可能な場合）
    # view(complex_TiO6)
    
    # 構造をファイルに保存
    complex_TiO6.write('TiO6_octahedral.xyz')
    print("構造をTiO6_octahedral.xyzに保存しました")
    

## 演習問題

**演習2.1：基本レベル - 結晶場分裂の計算**

**問題** ：Ni2+（d8）が八面体配位された場合、結晶場分裂エネルギーΔoct = 8,500 cm-1のとき、以下を求めよ。

  1. eg軌道とt2g軌道のエネルギー（cm-1）
  2. Crystal Field Stabilization Energy (CFSE)
  3. 高スピン状態と低スピン状態、どちらが安定か？

**ヒント** ：

  * eg軌道：+ 0.6 Δoct
  * t2g軌道：- 0.4 Δoct
  * Ni2+（d8）はほぼ常に高スピン状態

**演習2.2：中級レベル - Jahn-Teller効果**

**問題** ：以下の遷移金属イオンのうち、Jahn-Teller効果を示すものをすべて選び、理由を説明せよ。

  1. Ti3+（d1）
  2. Cr3+（d3）
  3. Mn3+（d4、高スピン）
  4. Fe2+（d6、高スピン）
  5. Cu2+（d9）

**Pythonで電子配置を図示せよ**

**演習2.3：中級レベル - 分光化学系列**

**問題** ：[Co(H2O)6]2+と[Co(NH3)6]2+の色が異なる理由を、分光化学系列を用いて説明せよ。

  * [Co(H2O)6]2+：ピンク色
  * [Co(NH3)6]2+：黄褐色

**Pythonで吸収スペクトルの違いを可視化せよ**

**演習2.4：応用レベル - Tanabe-Sugano図の読解**

**問題** ：d3電子配置を持つCr3+（ルビー）のTanabe-Sugano図から、以下を読み取れ。

  1. 基底状態の項記号
  2. 可視領域（15,000～25,000 cm-1）に現れる吸収帯とその起源
  3. Racahパラメータ B = 918 cm-1、Δoct = 17,400 cm-1のとき、Δ/Bを計算せよ

**Pythonで簡易Tanabe-Sugano図を描画し、吸収遷移を図示せよ**

**演習2.5：応用レベル - スピンクロスオーバー**

**問題** ：Fe2+（d6）錯体が高スピン状態（S=2）から低スピン状態（S=0）へ転移する条件を、以下のパラメータを用いて計算せよ。

  * Pairing energy: P = 15,000 cm-1
  * 結晶場分裂：Δoct = 10,000～25,000 cm-1

**Pythonでスピンクロスオーバー曲線を描画し、転移温度を推定せよ**

**演習2.6：応用レベル - DFT計算の準備**

**問題** ：MnO（岩塩型構造）の結晶場分裂をDFTで計算するため、以下を準備せよ。

  1. ASEを用いてMnOの単位格子を生成（格子定数 a = 4.445 Å）
  2. VASP入力ファイル（INCAR, POSCAR, KPOINTS）を作成
  3. DFT+U法のパラメータ（Ueff = 4.0 eV）を設定

**期待される結果** ：Mn2+のd軌道PDOSからΔoctを抽出

**演習2.7：統合演習 - 触媒材料の設計**

**問題** ：水分解触媒としての遷移金属酸化物を設計するため、以下を検討せよ。

  1. Co3+、Ni3+、Cu2+の中で、最も eg 軌道占有が適切なイオンは？
  2. 八面体配位と四面体配位、どちらが触媒活性に有利か？
  3. DFT計算で検証すべき物性パラメータを3つ挙げよ

**Pythonで候補材料のd軌道エネルギー図を比較せよ**

**演習2.8：研究プロジェクト - 新規磁性材料の探索**

**プロジェクト課題** ：Jahn-Teller効果を利用した新規強磁性材料を提案せよ。

**要求事項** ：

  1. 適切な遷移金属イオンの選定（d電子配置の根拠）
  2. 結晶構造と配位環境の設計
  3. DFT計算による磁気モーメントの予測
  4. Curie温度の概算
  5. 合成可能性の評価

**提出物** ：

  * 材料設計のレポート（2ページ）
  * Pythonコード（構造生成、エネルギー計算、可視化）
  * VASP入力ファイル一式

## まとめ

この章では、**結晶場理論** と**配位子場理論** を学び、遷移金属化合物のd軌道分裂と電子状態の関係を理解しました。

### 重要ポイントの再確認

  * ✅ **結晶場理論** ：配位子を点電荷として扱い、d軌道の分裂を静電的に説明
  * ✅ **八面体配位** ：egとt2gに分裂、Δoctが分裂幅
  * ✅ **四面体配位** ：分裂パターンが逆転、Δtet ≈ 4/9 Δoct
  * ✅ **Jahn-Teller効果** ：縮退した電子状態が構造歪みで解消される
  * ✅ **配位子場理論** ：共有結合性を考慮した精密な理論
  * ✅ **分光化学系列** ：配位子の結晶場分裂の強さの順序
  * ✅ **Tanabe-Sugano図** ：dn電子状態とΔ/Bの関係を図示
  * ✅ **応用** ：触媒、磁性材料、光学材料、色素

### 次章への接続

第3章では、これらの概念を**第一原理計算（DFT）** で実際に計算する方法を学びます。Hohenberg-Kohn定理、Kohn-Sham方程式、そしてPythonライブラリ（ASE, Pymatgen）を使った実践的ワークフローを習得しましょう。
