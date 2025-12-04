---
title: 第3章：結晶構造の基礎
chapter_title: 第3章：結晶構造の基礎
subtitle: 原子の規則的配列と材料特性の関係
reading_time: 30-35分
difficulty: 中級
code_examples: 7
---

材料の性質は、原子がどのように配列しているかに大きく依存します。この章では、結晶構造の基礎概念、主要な結晶系、そしてミラー指数について学び、Pythonで結晶構造を可視化します。 

## 学習目標

この章を読むことで、以下を習得できます：

  * ✅ 結晶と非晶質の違いを理解し、X線回折パターンから判別できる
  * ✅ 単位格子と格子定数の概念を理解する
  * ✅ 主要な結晶構造（FCC, BCC, HCP）の特徴と代表材料を説明できる
  * ✅ ミラー指数を使って結晶面と方向を表記できる
  * ✅ 充填率と配位数を計算できる
  * ✅ Pythonで3D結晶構造を可視化できる

* * *

## 3.1 結晶と非晶質

### 結晶構造とは

**結晶（Crystal）** とは、原子が3次元空間に規則的かつ周期的に配列した固体です。一方、**非晶質（Amorphous）** は、原子配列に長距離秩序がない固体です。

特性 | 結晶 | 非晶質  
---|---|---  
**原子配列** | 長距離秩序あり（周期的） | 短距離秩序のみ（ランダム）  
**融点** | 明確な融点あり | ガラス転移温度（徐々に軟化）  
**X線回折** | 鋭いピーク（ブラッグ反射） | ブロードなハロー  
**異方性** | 方向により性質が異なる | 等方性（全方向で同じ）  
**代表例** | 金属、Si、NaCl、ダイヤモンド | ガラス、高分子、アモルファスSi  
  
### X線回折による結晶性の評価

**X線回折（XRD: X-ray Diffraction）** は、結晶構造を解析する最も重要な手法です。X線が結晶に入射すると、原子配列により特定の角度で**ブラッグ反射** が起こります。

**ブラッグの法則** :

$$n\lambda = 2d\sin\theta$$

ここで、

  * $n$: 反射次数（整数）
  * $\lambda$: X線の波長
  * $d$: 結晶面間隔
  * $\theta$: 入射角（ブラッグ角）

結晶性材料では特定の角度（$2\theta$）で強い回折ピークが現れ、非晶質材料ではブロードなハローパターンが観察されます。

### コード例1: 結晶と非晶質のX線回折パターンシミュレーション

結晶性材料と非晶質材料のX線回折パターンの違いを可視化します。
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    # X線回折パターンのシミュレーション
    def simulate_crystalline_xrd():
        """
        結晶性材料のXRDパターンをシミュレート
        鋭いブラッグピークが特定の角度に出現
        """
        # 2θ範囲（度）
        two_theta = np.linspace(10, 80, 1000)
        intensity = np.zeros_like(two_theta)
    
        # 主要な結晶面からのブラッグ反射ピーク
        # (111), (200), (220), (311)面からの回折
        peaks = [28.4, 33.0, 47.5, 56.1]  # 2θ位置（例：FCC構造）
        peak_intensities = [100, 50, 80, 40]  # 相対強度
        peak_width = 0.3  # ピーク幅（度）
    
        # ガウシアンピークを追加
        for peak_pos, peak_int in zip(peaks, peak_intensities):
            intensity += peak_int * np.exp(-((two_theta - peak_pos) / peak_width)**2)
    
        # バックグラウンドノイズ
        intensity += np.random.normal(2, 0.5, len(two_theta))
    
        return two_theta, intensity
    
    
    def simulate_amorphous_xrd():
        """
        非晶質材料のXRDパターンをシミュレート
        ブロードなハローパターン
        """
        two_theta = np.linspace(10, 80, 1000)
    
        # ブロードなハロー（短距離秩序による）
        halo_center = 25  # ハローの中心位置
        halo_width = 15   # ハローの幅
        intensity = 30 * np.exp(-((two_theta - halo_center) / halo_width)**2)
    
        # 追加のブロードなピーク
        intensity += 15 * np.exp(-((two_theta - 45) / 20)**2)
    
        # ノイズ
        intensity += np.random.normal(2, 0.5, len(two_theta))
    
        return two_theta, intensity
    
    
    # プロット作成
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # 結晶性材料のXRDパターン
    two_theta_cryst, intensity_cryst = simulate_crystalline_xrd()
    ax1.plot(two_theta_cryst, intensity_cryst, linewidth=1.5, color='#1f77b4')
    ax1.set_xlabel('2θ (度)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('強度 (a.u.)', fontsize=12, fontweight='bold')
    ax1.set_title('結晶性材料のXRDパターン\n（鋭いブラッグピーク）', fontsize=13, fontweight='bold')
    ax1.grid(alpha=0.3)
    ax1.set_ylim(0, 110)
    
    # ピーク位置に注釈
    peaks_labels = ['(111)', '(200)', '(220)', '(311)']
    peaks_pos = [28.4, 33.0, 47.5, 56.1]
    for label, pos in zip(peaks_labels, peaks_pos):
        ax1.annotate(label, xy=(pos, 105), ha='center', fontsize=10,
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.5))
    
    # 非晶質材料のXRDパターン
    two_theta_amor, intensity_amor = simulate_amorphous_xrd()
    ax2.plot(two_theta_amor, intensity_amor, linewidth=1.5, color='#ff7f0e')
    ax2.set_xlabel('2θ (度)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('強度 (a.u.)', fontsize=12, fontweight='bold')
    ax2.set_title('非晶質材料のXRDパターン\n（ブロードなハロー）', fontsize=13, fontweight='bold')
    ax2.grid(alpha=0.3)
    ax2.set_ylim(0, 35)
    
    plt.tight_layout()
    plt.show()
    
    print("XRDパターンの解釈:")
    print("\n【結晶性材料】")
    print("- 鋭いピークが明確な角度に出現")
    print("- ピーク位置から格子定数と結晶構造を決定")
    print("- ピーク強度から原子配列の情報を取得")
    print("- 例：金属、セラミックス、シリコン単結晶")
    
    print("\n【非晶質材料】")
    print("- ブロードなハローパターン")
    print("- 短距離秩序は存在するが、長距離秩序なし")
    print("- ハローの位置から平均原子間距離を推定")
    print("- 例：ガラス、アモルファスシリコン、一部の高分子")
    

**解説** : X線回折パターンは結晶性の有無を明確に示します。結晶性材料は特定の角度で鋭いピークを示し、これにより結晶構造を同定できます。非晶質材料はブロードなパターンを示し、長距離秩序がないことがわかります。

* * *

## 3.2 単位格子と格子定数

### 単位格子（Unit Cell）

**単位格子** は、結晶構造の最小の繰り返し単位です。3次元空間に単位格子を並べることで、結晶全体を構成できます。

**格子定数（Lattice Parameters）** :

  * $a, b, c$: 単位格子の辺の長さ（Å単位）
  * $\alpha, \beta, \gamma$: 辺の間の角度（度またはラジアン）

### 7つの結晶系

単位格子の形状により、結晶は7つの結晶系に分類されます：

結晶系 | 格子定数の関係 | 角度の関係 | 代表例  
---|---|---|---  
**立方晶系** | a = b = c | α = β = γ = 90° | NaCl, Cu, Fe, Si  
**正方晶系** | a = b ≠ c | α = β = γ = 90° | TiO₂, SnO₂  
**斜方晶系** | a ≠ b ≠ c | α = β = γ = 90° | α-S, BaSO₄  
**六方晶系** | a = b ≠ c | α = β = 90°, γ = 120° | Mg, Zn, グラファイト  
**三方晶系** | a = b = c | α = β = γ ≠ 90° | 石英, CaCO₃  
**単斜晶系** | a ≠ b ≠ c | α = γ = 90° ≠ β | β-S, CaSO₄·2H₂O  
**三斜晶系** | a ≠ b ≠ c | α ≠ β ≠ γ ≠ 90° | CuSO₄·5H₂O  
  
材料科学で最も重要なのは**立方晶系** です。金属の多くは立方晶系に属します。

### ミラー指数（Miller Indices）

**ミラー指数** は、結晶中の面や方向を表記する方法です。

**結晶面の表記** : $(hkl)$

**結晶方向の表記** : $[uvw]$

**ミラー指数の決定方法** （結晶面の場合）:

  1. 面がa, b, c軸と交わる点の座標を求める（切片）
  2. 各切片の逆数を取る
  3. 整数比になるように通分する
  4. $(hkl)$と表記する

**例** :

  * $(100)$面: a軸に垂直な面
  * $(110)$面: a軸とb軸に対して45°傾いた面
  * $(111)$面: a, b, c軸に対して等しく傾いた面

### 面間隔の計算

立方晶系の場合、$(hkl)$面の面間隔$d_{hkl}$は以下の式で計算されます：

$$d_{hkl} = \frac{a}{\sqrt{h^2 + k^2 + l^2}}$$

ここで、$a$は格子定数です。

### コード例2: 単位格子体積と面間隔の計算機

格子定数から単位格子の体積と、ミラー指数で指定された面の面間隔を計算します。
    
    
    import numpy as np
    
    class CrystalCalculator:
        """
        結晶構造の各種パラメータを計算するクラス
        """
    
        def __init__(self, a, b=None, c=None, alpha=90, beta=90, gamma=90):
            """
            格子定数を設定
    
            Parameters:
            a, b, c: 格子定数 (Å)
            alpha, beta, gamma: 角度 (度)
            """
            self.a = a
            self.b = b if b is not None else a
            self.c = c if c is not None else a
            self.alpha = np.radians(alpha)
            self.beta = np.radians(beta)
            self.gamma = np.radians(gamma)
    
        def unit_cell_volume(self):
            """
            単位格子の体積を計算 (Å³)
            """
            # 一般式（全ての結晶系に対応）
            cos_alpha = np.cos(self.alpha)
            cos_beta = np.cos(self.beta)
            cos_gamma = np.cos(self.gamma)
    
            volume = self.a * self.b * self.c * np.sqrt(
                1 - cos_alpha**2 - cos_beta**2 - cos_gamma**2
                + 2*cos_alpha*cos_beta*cos_gamma
            )
            return volume
    
        def d_spacing_cubic(self, h, k, l):
            """
            立方晶系の面間隔を計算（簡易版）
    
            Parameters:
            h, k, l: ミラー指数
    
            Returns:
            d: 面間隔 (Å)
            """
            if h == 0 and k == 0 and l == 0:
                raise ValueError("ミラー指数が全て0になることはありません")
    
            d = self.a / np.sqrt(h**2 + k**2 + l**2)
            return d
    
        def print_crystal_info(self, crystal_name):
            """
            結晶情報を見やすく表示
            """
            print(f"\n{'='*60}")
            print(f"【{crystal_name}の結晶パラメータ】")
            print(f"{'='*60}")
            print(f"格子定数 a = {self.a:.4f} Å")
            if self.b != self.a or self.c != self.a:
                print(f"格子定数 b = {self.b:.4f} Å")
                print(f"格子定数 c = {self.c:.4f} Å")
    
            volume = self.unit_cell_volume()
            print(f"単位格子体積 V = {volume:.4f} Å³")
    
            # 主要な結晶面の面間隔を計算（立方晶系の場合）
            if self.a == self.b == self.c and \
               self.alpha == self.beta == self.gamma == np.radians(90):
                print(f"\n主要結晶面の面間隔:")
                planes = [(1,0,0), (1,1,0), (1,1,1), (2,0,0), (2,2,0)]
                for h, k, l in planes:
                    d = self.d_spacing_cubic(h, k, l)
                    print(f"  ({h}{k}{l})面: d = {d:.4f} Å")
    
    
    # 代表的な材料の結晶パラメータ
    print("代表的な材料の結晶構造計算")
    
    # 銅（FCC）
    cu = CrystalCalculator(a=3.615)  # Å
    cu.print_crystal_info("銅（Cu, FCC）")
    
    # 鉄（BCC）
    fe = CrystalCalculator(a=2.866)  # Å
    fe.print_crystal_info("鉄（Fe, BCC）")
    
    # シリコン（ダイヤモンド構造）
    si = CrystalCalculator(a=5.431)  # Å
    si.print_crystal_info("シリコン（Si）")
    
    # アルミナ（六方晶系）
    al2o3 = CrystalCalculator(a=4.759, c=12.991, gamma=120)  # Å
    print(f"\n{'='*60}")
    print(f"【アルミナ（Al₂O₃, HCP）の結晶パラメータ】")
    print(f"{'='*60}")
    print(f"格子定数 a = {al2o3.a:.4f} Å")
    print(f"格子定数 c = {al2o3.c:.4f} Å")
    print(f"c/a比 = {al2o3.c/al2o3.a:.4f}")
    volume = al2o3.unit_cell_volume()
    print(f"単位格子体積 V = {volume:.4f} Å³")
    
    print("\n" + "="*60)
    print("面間隔の応用:")
    print("- X線回折のピーク位置予測")
    print("- 原子面の密度計算")
    print("- すべり系の解析（塑性変形メカニズム）")
    

**出力例** :
    
    
    代表的な材料の結晶構造計算
    ============================================================
    【銅（Cu, FCC）の結晶パラメータ】
    ============================================================
    格子定数 a = 3.6150 Å
    単位格子体積 V = 47.2418 Å³
    
    主要結晶面の面間隔:
      (100)面: d = 3.6150 Å
      (110)面: d = 2.5557 Å
      (111)面: d = 2.0871 Å
      (200)面: d = 1.8075 Å
      (220)面: d = 1.2779 Å

**解説** : 格子定数から単位格子体積と面間隔を計算できます。面間隔はX線回折のピーク位置の予測や、原子面の密度計算に使われます。

* * *

## 3.3 主要な結晶構造

金属材料の多くは、以下の3つの結晶構造のいずれかを持ちます：

### 1\. 面心立方格子（FCC: Face-Centered Cubic）

**特徴** :

  * 立方体の各面の中心に原子が配置
  * 単位格子あたりの原子数: 4個（頂点8×1/8 + 面心6×1/2 = 4）
  * 配位数: 12（最も近い原子の数）
  * 充填率（APF: Atomic Packing Fraction）: 74%
  * 最密充填構造

**代表材料** :

  * 銅（Cu）: a = 3.615 Å
  * アルミニウム（Al）: a = 4.049 Å
  * 金（Au）: a = 4.078 Å
  * 銀（Ag）: a = 4.086 Å
  * ニッケル（Ni）: a = 3.524 Å

**性質** : 延性に優れる（多くのすべり系）、比較的柔らかい

### 2\. 体心立方格子（BCC: Body-Centered Cubic）

**特徴** :

  * 立方体の中心に原子が配置
  * 単位格子あたりの原子数: 2個（頂点8×1/8 + 体心1 = 2）
  * 配位数: 8
  * 充填率（APF）: 68%

**代表材料** :

  * 鉄（Fe, α-鉄）: a = 2.866 Å
  * クロム（Cr）: a = 2.885 Å
  * タングステン（W）: a = 3.165 Å
  * モリブデン（Mo）: a = 3.147 Å
  * バナジウム（V）: a = 3.024 Å

**性質** : 強度が高い、低温で脆性破壊しやすい

### 3\. 六方最密充填（HCP: Hexagonal Close-Packed）

**特徴** :

  * 六方晶系の最密充填構造
  * 単位格子あたりの原子数: 6個
  * 配位数: 12
  * 充填率（APF）: 74%（FCCと同じ）
  * 理想的なc/a比: 1.633

**代表材料** :

  * マグネシウム（Mg）: a = 3.209 Å, c/a = 1.624
  * 亜鉛（Zn）: a = 2.665 Å, c/a = 1.856
  * チタン（Ti）: a = 2.951 Å, c/a = 1.588
  * コバルト（Co）: a = 2.507 Å, c/a = 1.623

**性質** : すべり系が少なく、延性が低い（異方性が強い）

### 結晶構造の比較表

項目 | FCC | BCC | HCP  
---|---|---|---  
**原子数/単位格子** | 4 | 2 | 6  
**配位数** | 12 | 8 | 12  
**充填率（APF）** | 74% | 68% | 74%  
**すべり系の数** | 12 | 48（低温で制限） | 3（少ない）  
**延性** | 高 | 中〜高 | 低  
**代表金属** | Cu, Al, Au, Ag | Fe, Cr, W, Mo | Mg, Zn, Ti, Co  
  
### コード例3: 充填率（APF）の計算

FCC, BCC, HCP構造の原子充填率を計算し、比較します。
    
    
    import numpy as np
    
    def calculate_apf_fcc(a, r=None):
        """
        FCC構造の充填率（APF）を計算
    
        Parameters:
        a: 格子定数 (Å)
        r: 原子半径 (Å)。Noneの場合、格子定数から計算
    
        Returns:
        apf: 充填率
        """
        # FCC構造では、面対角線に沿って原子が接触
        # 面対角線 = 4r = a√2 より、r = a√2/4
        if r is None:
            r = a * np.sqrt(2) / 4
    
        # 単位格子あたりの原子数
        n_atoms = 4
    
        # 原子の体積
        v_atoms = n_atoms * (4/3) * np.pi * r**3
    
        # 単位格子の体積
        v_cell = a**3
    
        # 充填率
        apf = v_atoms / v_cell
    
        return apf, r
    
    
    def calculate_apf_bcc(a, r=None):
        """
        BCC構造の充填率（APF）を計算
    
        Parameters:
        a: 格子定数 (Å)
        r: 原子半径 (Å)。Noneの場合、格子定数から計算
    
        Returns:
        apf: 充填率
        """
        # BCC構造では、体対角線に沿って原子が接触
        # 体対角線 = 4r = a√3 より、r = a√3/4
        if r is None:
            r = a * np.sqrt(3) / 4
    
        # 単位格子あたりの原子数
        n_atoms = 2
    
        # 原子の体積
        v_atoms = n_atoms * (4/3) * np.pi * r**3
    
        # 単位格子の体積
        v_cell = a**3
    
        # 充填率
        apf = v_atoms / v_cell
    
        return apf, r
    
    
    def calculate_apf_hcp(a, c, r=None):
        """
        HCP構造の充填率（APF）を計算
    
        Parameters:
        a: 基底面の格子定数 (Å)
        c: c軸方向の格子定数 (Å)
        r: 原子半径 (Å)。Noneの場合、a/2と仮定
    
        Returns:
        apf: 充填率
        """
        # HCP構造では、基底面内で原子が接触
        # a = 2r
        if r is None:
            r = a / 2
    
        # 単位格子あたりの原子数
        n_atoms = 6
    
        # 原子の体積
        v_atoms = n_atoms * (4/3) * np.pi * r**3
    
        # 単位格子の体積（六方晶）
        # V = (√3/2) * a² * c
        v_cell = (np.sqrt(3) / 2) * a**2 * c
    
        # 充填率
        apf = v_atoms / v_cell
    
        return apf, r
    
    
    # 充填率の計算と比較
    print("="*70)
    print("主要な結晶構造の充填率（APF）計算")
    print("="*70)
    
    # FCC（銅を例）
    a_fcc = 3.615  # Å
    apf_fcc, r_fcc = calculate_apf_fcc(a_fcc)
    print(f"\n【FCC構造（例：銅）】")
    print(f"格子定数 a = {a_fcc} Å")
    print(f"原子半径 r = {r_fcc:.4f} Å")
    print(f"単位格子あたりの原子数 = 4")
    print(f"充填率 APF = {apf_fcc:.4f} ({apf_fcc*100:.2f}%)")
    print(f"理論値: 0.7405 (74.05%)")
    
    # BCC（鉄を例）
    a_bcc = 2.866  # Å
    apf_bcc, r_bcc = calculate_apf_bcc(a_bcc)
    print(f"\n【BCC構造（例：鉄）】")
    print(f"格子定数 a = {a_bcc} Å")
    print(f"原子半径 r = {r_bcc:.4f} Å")
    print(f"単位格子あたりの原子数 = 2")
    print(f"充填率 APF = {apf_bcc:.4f} ({apf_bcc*100:.2f}%)")
    print(f"理論値: 0.6802 (68.02%)")
    
    # HCP（マグネシウムを例、理想的なc/a比）
    a_hcp = 3.209  # Å
    c_hcp = a_hcp * np.sqrt(8/3)  # 理想的なc/a = 1.633
    apf_hcp, r_hcp = calculate_apf_hcp(a_hcp, c_hcp)
    print(f"\n【HCP構造（例：マグネシウム）】")
    print(f"格子定数 a = {a_hcp} Å")
    print(f"格子定数 c = {c_hcp:.4f} Å (理想値)")
    print(f"c/a比 = {c_hcp/a_hcp:.4f}")
    print(f"原子半径 r = {r_hcp:.4f} Å")
    print(f"単位格子あたりの原子数 = 6")
    print(f"充填率 APF = {apf_hcp:.4f} ({apf_hcp*100:.2f}%)")
    print(f"理論値: 0.7405 (74.05%)")
    
    # 比較
    print("\n" + "="*70)
    print("充填率の比較:")
    print("="*70)
    print(f"FCC: {apf_fcc*100:.2f}% - 最密充填、延性に優れる")
    print(f"BCC: {apf_bcc*100:.2f}% - やや疎、強度が高い")
    print(f"HCP: {apf_hcp*100:.2f}% - 最密充填、延性が低い")
    
    print("\n充填率と材料特性の関係:")
    print("- 高充填率 → 密度が高い、延性に優れる傾向")
    print("- 低充填率 → すき間が多い、原子の移動がしやすい")
    print("- 配位数が多い → より多くの結合、安定性が高い")
    

**出力例** :
    
    
    ======================================================================
    主要な結晶構造の充填率（APF）計算
    ======================================================================
    
    【FCC構造（例：銅）】
    格子定数 a = 3.615 Å
    原子半径 r = 1.2780 Å
    単位格子あたりの原子数 = 4
    充填率 APF = 0.7405 (74.05%)
    理論値: 0.7405 (74.05%)
    
    【BCC構造（例：鉄）】
    格子定数 a = 2.866 Å
    原子半径 r = 1.2410 Å
    単位格子あたりの原子数 = 2
    充填率 APF = 0.6802 (68.02%)
    理論値: 0.6802 (68.02%)

**解説** : 充填率（APF）は、単位格子の体積に対する原子が占める体積の割合です。FCCとHCPは74%で最密充填構造、BCCは68%でやや疎な構造です。充填率は材料の密度や機械的性質に影響します。

### コード例4: 配位数と最近接原子間距離の可視化

FCC, BCC構造の配位数（最近接原子の数）と原子間距離を可視化します。
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    def plot_coordination_fcc():
        """
        FCC構造の配位数を3Dプロットで可視化
        中心原子（面心）とその周囲の12個の最近接原子
        """
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
    
        # 格子定数（規格化: a=1）
        a = 1.0
    
        # 中心原子の位置（例：(0.5, 0.5, 0)面心）
        center = np.array([0.5, 0.5, 0])
    
        # FCC構造の最近接原子の相対位置（中心原子からの変位）
        # 配位数12: 4つの異なるタイプの面心との距離
        nearest_neighbors = [
            # 同一面内の4個
            np.array([0.5, 0, 0]), np.array([-0.5, 0, 0]),
            np.array([0, 0.5, 0]), np.array([0, -0.5, 0]),
            # 上面の4個
            np.array([0.5, 0, 1]), np.array([-0.5, 0, 1]),
            np.array([0, 0.5, 1]), np.array([0, -0.5, 1]),
            # 下面の4個
            np.array([0.5, 0, -1]), np.array([-0.5, 0, -1]),
            np.array([0, 0.5, -1]), np.array([0, -0.5, -1]),
        ]
    
        # 簡略化: 実際の位置を計算
        # 注：上記は概念的な説明用。実際には単位格子内の位置を正確に計算
    
        # より正確な最近接原子の位置（面心を中心として）
        # FCC: 面心から最も近い原子は頂点と他の面心
        neighbors_accurate = []
        for i in [-1, 1]:
            for j in [-1, 1]:
                neighbors_accurate.append(center + np.array([i*0.5, j*0.5, 0]))
                neighbors_accurate.append(center + np.array([i*0.5, 0, j*0.5]))
                neighbors_accurate.append(center + np.array([0, i*0.5, j*0.5]))
    
        # 重複除去
        neighbors_unique = []
        for n in neighbors_accurate:
            is_duplicate = False
            for nu in neighbors_unique:
                if np.allclose(n, nu):
                    is_duplicate = True
                    break
            if not is_duplicate:
                neighbors_unique.append(n)
    
        # 中心原子をプロット
        ax.scatter(*center, s=500, c='red', marker='o',
                   edgecolors='black', linewidth=2, label='中心原子', alpha=0.8)
    
        # 最近接原子をプロット
        neighbors_array = np.array(neighbors_unique)
        ax.scatter(neighbors_array[:, 0], neighbors_array[:, 1], neighbors_array[:, 2],
                   s=200, c='blue', marker='o', edgecolors='black', linewidth=1.5,
                   label='最近接原子', alpha=0.6)
    
        # 結合線を描画
        for neighbor in neighbors_unique:
            ax.plot([center[0], neighbor[0]],
                    [center[1], neighbor[1]],
                    [center[2], neighbor[2]],
                    'k--', linewidth=1, alpha=0.3)
    
        ax.set_xlabel('X', fontsize=12, fontweight='bold')
        ax.set_ylabel('Y', fontsize=12, fontweight='bold')
        ax.set_zlabel('Z', fontsize=12, fontweight='bold')
        ax.set_title('FCC構造の配位数（12）', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.set_box_aspect([1,1,1])
    
        return fig
    
    
    def calculate_nearest_neighbor_distances():
        """
        FCC, BCC, HCP構造の最近接原子間距離を計算
        """
        print("="*70)
        print("最近接原子間距離の計算")
        print("="*70)
    
        # FCC
        a_fcc = 3.615  # 銅の格子定数（Å）
        # FCC: 面対角線 = 4r = a√2 → 最近接距離 = 2r = a/√2
        d_fcc = a_fcc / np.sqrt(2)
        print(f"\n【FCC（銅）】")
        print(f"格子定数 a = {a_fcc} Å")
        print(f"最近接原子間距離 = {d_fcc:.4f} Å")
        print(f"配位数 = 12")
    
        # BCC
        a_bcc = 2.866  # 鉄の格子定数（Å）
        # BCC: 体対角線 = 4r = a√3 → 最近接距離 = 2r = a√3/2
        d_bcc = a_bcc * np.sqrt(3) / 2
        print(f"\n【BCC（鉄）】")
        print(f"格子定数 a = {a_bcc} Å")
        print(f"最近接原子間距離 = {d_bcc:.4f} Å")
        print(f"配位数 = 8")
    
        # HCP
        a_hcp = 3.209  # マグネシウムの格子定数（Å）
        # HCP: 基底面内 = a = 2r → 最近接距離 = a
        d_hcp = a_hcp
        print(f"\n【HCP（マグネシウム）】")
        print(f"格子定数 a = {a_hcp} Å")
        print(f"最近接原子間距離 = {d_hcp:.4f} Å")
        print(f"配位数 = 12")
    
        # 比較グラフ
        structures = ['FCC\n(Cu)', 'BCC\n(Fe)', 'HCP\n(Mg)']
        distances = [d_fcc, d_bcc, d_hcp]
        coordination_numbers = [12, 8, 12]
    
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
        # 最近接距離の比較
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        ax1.bar(structures, distances, color=colors, edgecolor='black', linewidth=1.5, alpha=0.7)
        ax1.set_ylabel('最近接原子間距離 (Å)', fontsize=12, fontweight='bold')
        ax1.set_title('結晶構造別の最近接原子間距離', fontsize=13, fontweight='bold')
        ax1.grid(axis='y', alpha=0.3)
    
        # 値をバーの上に表示
        for i, (s, d) in enumerate(zip(structures, distances)):
            ax1.text(i, d + 0.05, f'{d:.3f} Å', ha='center', fontsize=11, fontweight='bold')
    
        # 配位数の比較
        ax2.bar(structures, coordination_numbers, color=colors, edgecolor='black', linewidth=1.5, alpha=0.7)
        ax2.set_ylabel('配位数', fontsize=12, fontweight='bold')
        ax2.set_title('結晶構造別の配位数', fontsize=13, fontweight='bold')
        ax2.set_ylim(0, 14)
        ax2.grid(axis='y', alpha=0.3)
    
        # 値をバーの上に表示
        for i, (s, cn) in enumerate(zip(structures, coordination_numbers)):
            ax2.text(i, cn + 0.3, f'{cn}', ha='center', fontsize=12, fontweight='bold')
    
        plt.tight_layout()
        plt.show()
    
        print("\n" + "="*70)
        print("配位数と材料特性の関係:")
        print("- 配位数が多い → 結合が多い → 安定、密に詰まっている")
        print("- FCC, HCP: 配位数12 → 最密充填、延性に優れる")
        print("- BCC: 配位数8 → やや疎、強度が高いが低温で脆い")
    
    
    # 実行
    calculate_nearest_neighbor_distances()
    # plot_coordination_fcc()  # 3Dプロットが必要な場合はコメント解除
    plt.show()
    

**解説** : 配位数は、ある原子の周りに最も近接して配置されている原子の数です。配位数が多いほど、原子間の結合が多く、材料は安定で密に詰まっています。FCCとHCPは配位数12で最密充填、BCCは配位数8でやや疎な構造です。

* * *

## 3.4 結晶構造と材料特性

### 密度の計算

結晶構造から材料の理論密度を計算できます：

$$\rho = \frac{n \cdot M}{V_{cell} \cdot N_A}$$

ここで、

  * $\rho$: 密度（g/cm³）
  * $n$: 単位格子あたりの原子数
  * $M$: 原子量（g/mol）
  * $V_{cell}$: 単位格子の体積（cm³）
  * $N_A$: アボガドロ数（6.022×10²³ mol⁻¹）

### すべり系と延性

**すべり系（Slip System）** は、塑性変形時に原子面がすべる面と方向の組み合わせです。すべり系が多いほど、材料は延性に優れます。

**主要な結晶構造のすべり系** :

  * **FCC** : {111}⟨110⟩ → 4面 × 3方向 = 12すべり系 → 延性大
  * **BCC** : {110}⟨111⟩, {112}⟨111⟩, {123}⟨111⟩ → 48すべり系（理論上）→ 延性中
  * **HCP** : {0001}⟨11̄20⟩ → 1面 × 3方向 = 3すべり系 → 延性小

**結晶構造と機械的性質の関係** :

  * FCC金属（Cu, Al, Au）: 多くのすべり系 → 延性大、加工性良好
  * BCC金属（Fe, Cr, W）: 温度依存性大 → 低温で脆性、高温で延性
  * HCP金属（Mg, Zn, Ti）: すべり系少 → 延性小、加工困難

### コード例5: 密度計算ツール

結晶構造パラメータから材料の理論密度を計算します。
    
    
    import numpy as np
    
    # アボガドロ数
    NA = 6.022e23  # mol^-1
    
    def calculate_density(n_atoms, atomic_mass, a, b=None, c=None,
                         alpha=90, beta=90, gamma=90):
        """
        結晶構造から理論密度を計算
    
        Parameters:
        n_atoms: 単位格子あたりの原子数
        atomic_mass: 原子量 (g/mol)
        a, b, c: 格子定数 (Å)
        alpha, beta, gamma: 角度 (度)
    
        Returns:
        density: 密度 (g/cm³)
        """
        # 格子定数のデフォルト値設定
        if b is None:
            b = a
        if c is None:
            c = a
    
        # 角度をラジアンに変換
        alpha_rad = np.radians(alpha)
        beta_rad = np.radians(beta)
        gamma_rad = np.radians(gamma)
    
        # 単位格子体積の計算（Å³）
        cos_alpha = np.cos(alpha_rad)
        cos_beta = np.cos(beta_rad)
        cos_gamma = np.cos(gamma_rad)
    
        V_cell = a * b * c * np.sqrt(
            1 - cos_alpha**2 - cos_beta**2 - cos_gamma**2
            + 2*cos_alpha*cos_beta*cos_gamma
        )
    
        # Å³ → cm³に変換（1 Å = 10^-8 cm）
        V_cell_cm3 = V_cell * 1e-24
    
        # 密度計算
        density = (n_atoms * atomic_mass) / (V_cell_cm3 * NA)
    
        return density
    
    
    # 代表的な材料の理論密度を計算
    print("="*70)
    print("結晶構造から計算した理論密度")
    print("="*70)
    
    materials = [
        {
            'name': '銅（Cu, FCC）',
            'n_atoms': 4,
            'atomic_mass': 63.546,  # g/mol
            'a': 3.615,  # Å
            'structure': 'FCC'
        },
        {
            'name': '鉄（Fe, BCC）',
            'n_atoms': 2,
            'atomic_mass': 55.845,
            'a': 2.866,
            'structure': 'BCC'
        },
        {
            'name': 'アルミニウム（Al, FCC）',
            'n_atoms': 4,
            'atomic_mass': 26.982,
            'a': 4.049,
            'structure': 'FCC'
        },
        {
            'name': 'タングステン（W, BCC）',
            'n_atoms': 2,
            'atomic_mass': 183.84,
            'a': 3.165,
            'structure': 'BCC'
        },
        {
            'name': 'マグネシウム（Mg, HCP）',
            'n_atoms': 6,
            'atomic_mass': 24.305,
            'a': 3.209,
            'c': 5.211,
            'gamma': 120,
            'structure': 'HCP'
        },
        {
            'name': '金（Au, FCC）',
            'n_atoms': 4,
            'atomic_mass': 196.967,
            'a': 4.078,
            'structure': 'FCC'
        }
    ]
    
    # 密度計算と表示
    calculated_densities = []
    experimental_densities = [8.96, 7.87, 2.70, 19.25, 1.74, 19.32]  # g/cm³（実測値）
    
    for i, mat in enumerate(materials):
        if 'c' in mat:
            density = calculate_density(mat['n_atoms'], mat['atomic_mass'],
                                       mat['a'], c=mat['c'], gamma=mat.get('gamma', 90))
        else:
            density = calculate_density(mat['n_atoms'], mat['atomic_mass'], mat['a'])
    
        calculated_densities.append(density)
    
        print(f"\n【{mat['name']}】")
        print(f"結晶構造: {mat['structure']}")
        print(f"単位格子あたりの原子数: {mat['n_atoms']}")
        print(f"原子量: {mat['atomic_mass']} g/mol")
        print(f"格子定数 a = {mat['a']} Å" + (f", c = {mat['c']} Å" if 'c' in mat else ""))
        print(f"計算密度: {density:.3f} g/cm³")
        print(f"実測密度: {experimental_densities[i]:.3f} g/cm³")
        print(f"誤差: {abs(density - experimental_densities[i]) / experimental_densities[i] * 100:.2f}%")
    
    # 比較グラフ
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(materials))
    width = 0.35
    
    rects1 = ax.bar(x - width/2, calculated_densities, width, label='計算密度',
                    color='#1f77b4', edgecolor='black', linewidth=1.5, alpha=0.7)
    rects2 = ax.bar(x + width/2, experimental_densities, width, label='実測密度',
                    color='#ff7f0e', edgecolor='black', linewidth=1.5, alpha=0.7)
    
    ax.set_ylabel('密度 (g/cm³)', fontsize=13, fontweight='bold')
    ax.set_title('結晶構造から計算した密度と実測値の比較', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([m['name'].split('（')[0] for m in materials], rotation=15, ha='right')
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    
    # 値をバーの上に表示
    for rect1, rect2 in zip(rects1, rects2):
        height1 = rect1.get_height()
        height2 = rect2.get_height()
        ax.text(rect1.get_x() + rect1.get_width()/2., height1,
                f'{height1:.2f}', ha='center', va='bottom', fontsize=8)
        ax.text(rect2.get_x() + rect2.get_width()/2., height2,
                f'{height2:.2f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.show()
    
    print("\n" + "="*70)
    print("密度計算の応用:")
    print("- 材料の同定（XRDパターンと密度から結晶構造を決定）")
    print("- 欠陥や不純物の評価（実測密度と理論密度の差から空孔濃度を推定）")
    print("- 軽量化設計（密度と強度のバランス最適化）")
    

**出力例** :
    
    
    ======================================================================
    結晶構造から計算した理論密度
    ======================================================================
    
    【銅（Cu, FCC）】
    結晶構造: FCC
    単位格子あたりの原子数: 4
    原子量: 63.546 g/mol
    格子定数 a = 3.615 Å
    計算密度: 8.933 g/cm³
    実測密度: 8.960 g/cm³
    誤差: 0.30%
    
    【鉄（Fe, BCC）】
    結晶構造: BCC
    単位格子あたりの原子数: 2
    原子量: 55.845 g/mol
    格子定数 a = 2.866 Å
    計算密度: 7.879 g/cm³
    実測密度: 7.870 g/cm³
    誤差: 0.11%

**解説** : 結晶構造パラメータから理論密度を高精度で計算できます。計算値と実測値の差から、空孔などの欠陥濃度を推定できます。

### コード例6: すべり系の可視化

FCC構造の主要なすべり系{111}⟨110⟩を2Dで可視化します。
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon, FancyArrowPatch
    
    def visualize_slip_systems_fcc():
        """
        FCC構造のすべり系を2D投影で可視化
        {111}<110>すべり系
        """
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
        # (111)面の投影を描画
        # FCC構造の単位格子（簡略化した2D表現）
    
        # すべり面とすべり方向の例
        slip_systems = [
            {
                'plane': '(111)',
                'direction': '[1̄10]',
                'description': '最も活動的なすべり系'
            },
            {
                'plane': '(1̄11)',
                'direction': '[110]',
                'description': '対称的なすべり系'
            },
            {
                'plane': '(11̄1)',
                'direction': '[101]',
                'description': '追加のすべり系'
            }
        ]
    
        for idx, (ax, slip_sys) in enumerate(zip(axes, slip_systems)):
            # 原子位置（簡略化した2D配列）
            atoms_x = [0, 1, 2, 0.5, 1.5, 1]
            atoms_y = [0, 0, 0, 0.866, 0.866, 1.732]
    
            # 原子をプロット
            ax.scatter(atoms_x, atoms_y, s=300, c='lightblue',
                      edgecolors='black', linewidth=2, zorder=3)
    
            # すべり面を表示（灰色の帯）
            slip_plane = Polygon([(0, 0), (2, 0), (2.5, 0.866), (0.5, 0.866)],
                                alpha=0.2, facecolor='gray', edgecolor='black',
                                linewidth=1.5, linestyle='--', zorder=1)
            ax.add_patch(slip_plane)
    
            # すべり方向を矢印で表示
            arrow = FancyArrowPatch((0.5, 0.4), (1.8, 0.4),
                                   arrowstyle='->', mutation_scale=30,
                                   linewidth=3, color='red', zorder=2)
            ax.add_patch(arrow)
    
            ax.set_xlim(-0.5, 3)
            ax.set_ylim(-0.5, 2.5)
            ax.set_aspect('equal')
            ax.set_title(f'{slip_sys["plane"]} 面\n{slip_sys["direction"]} 方向\n({slip_sys["description"]})',
                        fontsize=11, fontweight='bold')
            ax.axis('off')
    
            # 凡例
            ax.text(0.1, 2.2, '● 原子', fontsize=9)
            ax.text(0.1, 2.0, '   すべり面', fontsize=9, color='gray')
            ax.text(0.1, 1.8, '→ すべり方向', fontsize=9, color='red')
    
        plt.suptitle('FCC構造のすべり系 {111}⟨110⟩（合計12系）',
                    fontsize=14, fontweight='bold', y=0.98)
        plt.tight_layout()
        plt.show()
    
        # すべり系の数と延性の関係を説明
        print("="*70)
        print("すべり系と材料の延性")
        print("="*70)
    
        print("\n【FCC構造】")
        print("- すべり面: {111}（4面）")
        print("- すべり方向: ⟨110⟩（各面に3方向）")
        print("- 合計すべり系: 4 × 3 = 12")
        print("- 延性: 非常に高い（多くのすべり系）")
        print("- 代表例: Cu, Al, Au, Ag → 容易に塑性変形")
    
        print("\n【BCC構造】")
        print("- すべり面: {110}, {112}, {123}（複数）")
        print("- すべり方向: ⟨111⟩")
        print("- 合計すべり系: 48（理論上）")
        print("- 延性: 温度依存性大（低温で制限される）")
        print("- 代表例: Fe, Cr, W → 低温で脆性破壊しやすい")
    
        print("\n【HCP構造】")
        print("- すべり面: {0001}（1面、基底面）")
        print("- すべり方向: ⟨112̄0⟩（3方向）")
        print("- 合計すべり系: 1 × 3 = 3（非常に少ない）")
        print("- 延性: 低い（すべり系が限定的）")
        print("- 代表例: Mg, Zn, Ti → 常温加工困難")
    
        print("\n" + "="*70)
        print("すべり系の多さと加工性:")
        print("- すべり系が多い → 塑性変形しやすい → 延性大、加工性良")
        print("- すべり系が少ない → 塑性変形困難 → 延性小、脆性破壊しやすい")
        print("- 材料選択: 加工が必要な用途ではFCC金属が有利")
    
    
    # 実行
    visualize_slip_systems_fcc()
    

**解説** : すべり系は、塑性変形時に原子面がすべる面と方向の組み合わせです。すべり系が多いほど、材料は様々な方向から力を受けても塑性変形でき、延性に優れます。FCCは12すべり系で延性大、HCPは3すべり系で延性小です。

### コード例7: 3D結晶構造の可視化（FCC, BCC, HCP）

主要な3つの結晶構造を3Dで可視化し、構造の違いを理解します。
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    def plot_crystal_structure_3d(structure_type='FCC'):
        """
        結晶構造を3Dで可視化
    
        Parameters:
        structure_type: 'FCC', 'BCC', 'HCP'のいずれか
        """
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
    
        # 格子定数（規格化: a=1）
        a = 1.0
    
        if structure_type == 'FCC':
            # FCC構造の原子位置
            positions = [
                # 頂点の8個（各1/8）
                [0, 0, 0], [a, 0, 0], [a, a, 0], [0, a, 0],
                [0, 0, a], [a, 0, a], [a, a, a], [0, a, a],
                # 面心の6個（各1/2）
                [a/2, a/2, 0], [a/2, 0, a/2], [0, a/2, a/2],
                [a/2, a/2, a], [a/2, a, a/2], [a, a/2, a/2]
            ]
            title = 'FCC（面心立方格子）'
            description = '単位格子あたり4原子\n配位数12、充填率74%'
    
        elif structure_type == 'BCC':
            # BCC構造の原子位置
            positions = [
                # 頂点の8個（各1/8）
                [0, 0, 0], [a, 0, 0], [a, a, 0], [0, a, 0],
                [0, 0, a], [a, 0, a], [a, a, a], [0, a, a],
                # 体心の1個
                [a/2, a/2, a/2]
            ]
            title = 'BCC（体心立方格子）'
            description = '単位格子あたり2原子\n配位数8、充填率68%'
    
        elif structure_type == 'HCP':
            # HCP構造の原子位置（簡略化した表現）
            c_a_ratio = 1.633  # 理想的なc/a比
            c = a * c_a_ratio
    
            # 基底面の3個 + 頂点の3個 + 内部の2個
            positions = [
                # 下層基底面（3個の角）
                [0, 0, 0], [a, 0, 0], [a/2, a*np.sqrt(3)/2, 0],
                # 上層基底面
                [0, 0, c], [a, 0, c], [a/2, a*np.sqrt(3)/2, c],
                # 内部（2個）
                [a/2, a/(2*np.sqrt(3)), c/2], [a/2, a*np.sqrt(3)/6, c/2]
            ]
            title = 'HCP（六方最密充填）'
            description = '単位格子あたり6原子\n配位数12、充填率74%'
        else:
            raise ValueError("structure_typeは'FCC', 'BCC', 'HCP'のいずれか")
    
        # 原子位置を配列に変換
        positions = np.array(positions)
    
        # 原子をプロット
        ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
                  s=300, c='lightblue', edgecolors='darkblue', linewidth=2,
                  alpha=0.8, depthshade=True)
    
        # 単位格子の枠を描画
        # 立方体の辺
        edges = [
            # 底面
            [[0, 0, 0], [a, 0, 0]], [[a, 0, 0], [a, a, 0]],
            [[a, a, 0], [0, a, 0]], [[0, a, 0], [0, 0, 0]],
            # 上面
            [[0, 0, a], [a, 0, a]], [[a, 0, a], [a, a, a]],
            [[a, a, a], [0, a, a]], [[0, a, a], [0, 0, a]],
            # 縦の辺
            [[0, 0, 0], [0, 0, a]], [[a, 0, 0], [a, 0, a]],
            [[a, a, 0], [a, a, a]], [[0, a, 0], [0, a, a]]
        ]
    
        if structure_type != 'HCP':
            for edge in edges:
                edge = np.array(edge)
                ax.plot3D(edge[:, 0], edge[:, 1], edge[:, 2],
                         'k-', linewidth=1.5, alpha=0.6)
        else:
            # HCPの場合は六角柱の枠
            # 簡略化のため省略（実装は複雑）
            pass
    
        # 軸ラベル
        ax.set_xlabel('X', fontsize=12, fontweight='bold')
        ax.set_ylabel('Y', fontsize=12, fontweight='bold')
        ax.set_zlabel('Z', fontsize=12, fontweight='bold')
        ax.set_title(f'{title}\n{description}', fontsize=13, fontweight='bold', pad=20)
    
        # 視点の調整
        ax.view_init(elev=20, azim=45)
        ax.set_box_aspect([1,1,1])
    
        # グリッドをオフ
        ax.grid(False)
    
        return fig, ax
    
    
    # 3つの結晶構造を並べて表示
    fig = plt.figure(figsize=(18, 6))
    
    for i, structure in enumerate(['FCC', 'BCC', 'HCP'], 1):
        ax = fig.add_subplot(1, 3, i, projection='3d')
    
        # 格子定数
        a = 1.0
    
        if structure == 'FCC':
            positions = np.array([
                [0, 0, 0], [a, 0, 0], [a, a, 0], [0, a, 0],
                [0, 0, a], [a, 0, a], [a, a, a], [0, a, a],
                [a/2, a/2, 0], [a/2, 0, a/2], [0, a/2, a/2],
                [a/2, a/2, a], [a/2, a, a/2], [a, a/2, a/2]
            ])
            title = 'FCC'
            info = '原子数: 4\n配位数: 12\nAPF: 74%'
    
        elif structure == 'BCC':
            positions = np.array([
                [0, 0, 0], [a, 0, 0], [a, a, 0], [0, a, 0],
                [0, 0, a], [a, 0, a], [a, a, a], [0, a, a],
                [a/2, a/2, a/2]
            ])
            title = 'BCC'
            info = '原子数: 2\n配位数: 8\nAPF: 68%'
    
        else:  # HCP
            c = a * 1.633
            positions = np.array([
                [0, 0, 0], [a, 0, 0], [a/2, a*np.sqrt(3)/2, 0],
                [0, 0, c], [a, 0, c], [a/2, a*np.sqrt(3)/2, c],
                [a/2, a/(2*np.sqrt(3)), c/2]
            ])
            title = 'HCP'
            info = '原子数: 6\n配位数: 12\nAPF: 74%'
    
        # 原子をプロット
        ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
                  s=250, c='lightblue', edgecolors='darkblue', linewidth=2,
                  alpha=0.8, depthshade=True)
    
        # 単位格子の枠
        if structure != 'HCP':
            edges = [
                [[0, 0, 0], [a, 0, 0]], [[a, 0, 0], [a, a, 0]],
                [[a, a, 0], [0, a, 0]], [[0, a, 0], [0, 0, 0]],
                [[0, 0, a], [a, 0, a]], [[a, 0, a], [a, a, a]],
                [[a, a, a], [0, a, a]], [[0, a, a], [0, 0, a]],
                [[0, 0, 0], [0, 0, a]], [[a, 0, 0], [a, 0, a]],
                [[a, a, 0], [a, a, a]], [[0, a, 0], [0, a, a]]
            ]
            for edge in edges:
                edge = np.array(edge)
                ax.plot3D(edge[:, 0], edge[:, 1], edge[:, 2],
                         'k-', linewidth=1.5, alpha=0.5)
    
        ax.set_xlabel('X', fontsize=10)
        ax.set_ylabel('Y', fontsize=10)
        ax.set_zlabel('Z', fontsize=10)
        ax.set_title(f'{title}\n{info}', fontsize=12, fontweight='bold')
        ax.view_init(elev=20, azim=45)
        ax.set_box_aspect([1,1,1])
        ax.grid(False)
    
    plt.suptitle('主要な結晶構造の3D可視化', fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.show()
    
    print("="*70)
    print("結晶構造可視化の意義:")
    print("="*70)
    print("\n3D可視化により以下を理解:")
    print("- 原子の空間配置（どこに原子が位置するか）")
    print("- 充填のされ方（密に詰まっているか、すき間があるか）")
    print("- 対称性（構造の美しさと物性の関係）")
    print("- すべり面の予測（どの面がすべりやすいか）")
    
    print("\n実際の材料開発での活用:")
    print("- 新材料の結晶構造予測")
    print("- 相変態の理解（FCC→BCCなど）")
    print("- 材料特性の予測（構造から性質を推定）")
    

**解説** : 3D可視化により、FCC, BCC, HCPの原子配置の違いが明確になります。FCCとBCCは立方晶系、HCPは六方晶系であり、原子の詰まり方（充填率）や配位数が異なります。これらの違いが材料の機械的性質に大きく影響します。

* * *

## 3.5 本章のまとめ

### 学んだこと

  1. **結晶と非晶質の違い**
     * 結晶: 長距離秩序あり、明確な融点、鋭いXRDピーク
     * 非晶質: 短距離秩序のみ、ガラス転移、ブロードなXRDパターン
  2. **単位格子と格子定数**
     * 単位格子: 結晶の最小繰り返し単位
     * 7つの結晶系: 立方、正方、斜方、六方、三方、単斜、三斜
     * ミラー指数: (hkl)で結晶面、[uvw]で方向を表記
  3. **主要な結晶構造**
     * FCC: 原子数4、配位数12、APF 74%、延性大（Cu, Al, Au）
     * BCC: 原子数2、配位数8、APF 68%、強度高（Fe, Cr, W）
     * HCP: 原子数6、配位数12、APF 74%、延性小（Mg, Zn, Ti）
  4. **結晶構造と材料特性**
     * 密度は結晶構造から計算可能
     * すべり系の多さが延性を決定
     * FCC: 12すべり系 → 延性大
     * HCP: 3すべり系 → 延性小
  5. **Pythonによる可視化**
     * XRDパターンのシミュレーション
     * 充填率と配位数の計算
     * 密度計算ツール
     * 3D結晶構造の可視化

### 重要なポイント

  * 結晶構造は**材料の機械的性質を支配する** 重要因子
  * 充填率が高いほど、一般に密度が高く、延性に優れる
  * すべり系の数が延性を決定（多いほど延性大）
  * XRDは結晶構造解析の最も重要な手法
  * ミラー指数は結晶面と方向を表記する標準的な方法

### 次の章へ

第4章では、**材料の性質と構造の関係** を学びます：

  * 機械的性質（応力-ひずみ曲線、硬度）
  * 電気的性質（バンド構造、導電性）
  * 熱的性質（熱伝導率、熱膨張）
  * 光学的性質（吸収スペクトル、色）
  * Pythonによる材料特性の計算とプロット
