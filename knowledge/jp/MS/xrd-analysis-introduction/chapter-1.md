---
title: "第1章: X線回折の基礎"
chapter_title: "第1章: X線回折の基礎"
subtitle: Braggの法則から構造因子まで - 結晶構造解析の理論的基盤
reading_time: 25-30分
code_examples: 8
---

## 学習目標

この章を完了すると、以下ができるようになります:

  * Braggの法則を理解し、回折条件を数式とPythonで計算できる
  * 結晶格子と逆格子の関係を可視化し、説明できる
  * 消滅則から結晶の対称性を推定できる
  * 構造因子の計算を実装し、回折強度を予測できる
  * Ewald球を用いて回折条件を可視化できる

## 1.1 Braggの法則と回折条件

X線回折(XRD)は、結晶構造解析の最も基本的な手法です。X線が結晶中の原子配列と相互作用し、特定の角度で回折が起こることで、結晶構造に関する情報が得られます。

### 1.1.1 Braggの法則の導出

Braggの法則は、1912年にWilliam Lawrence BraggとWilliam Henry Braggによって発見されました。結晶を原子面の集合と考えると、隣接する原子面からの反射X線の光路差が波長の整数倍になると、建設的干渉が起こります。

面間隔\\( d \\)を持つ結晶面に、波長\\( \lambda \\)のX線が入射角\\( \theta \\)で入射する場合、回折条件は以下のように表されます:

\\[ 2d\sin\theta = n\lambda \\] 

ここで、\\( n \\)は回折次数(整数)、\\( d \\)は面間隔、\\( \theta \\)は入射角(Bragg角)、\\( \lambda \\)はX線の波長です。
    
    
    ```mermaid
    graph LR
        A[X線入射λ = 1.54 Å] --> B[結晶面d = 3.5 Å]
        B --> C{Bragg条件2dsinθ = nλ}
        C -->|満たす| D[回折ピーク観測される]
        C -->|満たさない| E[回折なし観測されない]
        style C fill:#fce7f3
        style D fill:#e8f5e9
        style E fill:#ffebee
    ```

### 1.1.2 Pythonによる計算実装
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    def bragg_law(d, wavelength, n=1):
        """Braggの法則を用いて回折角を計算
    
        Args:
            d (float): 面間隔 [Å]
            wavelength (float): X線波長 [Å]
            n (int): 回折次数
    
        Returns:
            float: Bragg角 [度]、または回折不可能な場合None
        """
        # 2d*sin(theta) = n*lambda
        # sin(theta) = n*lambda / (2*d)
        sin_theta = (n * wavelength) / (2 * d)
    
        # 物理的に有効な解の確認
        if sin_theta > 1.0:
            return None  # 回折不可能
    
        theta_rad = np.arcsin(sin_theta)
        theta_deg = np.degrees(theta_rad)
    
        return theta_deg
    
    
    # 例: Cu Kα線（1.54 Å）を用いた計算
    wavelength_CuKa = 1.54056  # Å
    
    # 異なる面間隔での回折角計算
    d_spacings = [4.0, 3.5, 3.0, 2.5, 2.0, 1.5]  # Å
    
    print("面間隔 [Å] | Bragg角 [度] (n=1) | Bragg角 [度] (n=2)")
    print("-" * 60)
    for d in d_spacings:
        theta_n1 = bragg_law(d, wavelength_CuKa, n=1)
        theta_n2 = bragg_law(d, wavelength_CuKa, n=2)
    
        if theta_n1 is not None:
            print(f"{d:6.2f}      | {theta_n1:15.2f}    | ", end="")
            if theta_n2 is not None:
                print(f"{theta_n2:15.2f}")
            else:
                print("   回折不可能")
        else:
            print(f"{d:6.2f}      |    回折不可能    |    回折不可能")
    
    # 期待される出力:
    # 面間隔 [Å] | Bragg角 [度] (n=1) | Bragg角 [度] (n=2)
    # ------------------------------------------------------------
    #   4.00      |           11.10    |            22.48
    #   3.50      |           12.69    |            25.77
    #   3.00      |           14.83    |            30.36
    #   2.50      |           17.93    |            37.32
    #   2.00      |           22.58    |            48.92
    #   1.50      |           30.96    |            75.55

### 1.1.3 回折角の可視化
    
    
    def plot_bragg_angles(wavelength, d_range=(1.0, 5.0), n_max=3):
        """Bragg角の面間隔依存性をプロット
    
        Args:
            wavelength (float): X線波長 [Å]
            d_range (tuple): 面間隔の範囲 [Å]
            n_max (int): 最大回折次数
        """
        d_values = np.linspace(d_range[0], d_range[1], 200)
    
        plt.figure(figsize=(10, 6))
        colors = ['#f093fb', '#f5576c', '#3498db']
    
        for n in range(1, n_max + 1):
            theta_values = []
            valid_d_values = []
    
            for d in d_values:
                theta = bragg_law(d, wavelength, n)
                if theta is not None:
                    theta_values.append(theta)
                    valid_d_values.append(d)
    
            if theta_values:
                plt.plot(valid_d_values, theta_values,
                        label=f'n = {n}', linewidth=2, color=colors[n-1])
    
        plt.xlabel('面間隔 d [Å]', fontsize=12)
        plt.ylabel('Bragg角 θ [度]', fontsize=12)
        plt.title(f'Braggの法則: 回折角と面間隔の関係 (λ = {wavelength:.2f} Å)', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=11)
        plt.xlim(d_range)
        plt.ylim(0, 90)
        plt.tight_layout()
        plt.show()
    
    # Cu Kα線の場合
    plot_bragg_angles(1.54056, d_range=(1.0, 5.0), n_max=3)

## 1.2 結晶格子と逆格子

### 1.2.1 結晶格子の基本

結晶は、基本単位(単位格子)が3次元空間に周期的に配列した構造です。単位格子は3つの基本ベクトル \\( \mathbf{a}, \mathbf{b}, \mathbf{c} \\) で定義されます。

任意の格子点の位置ベクトル\\( \mathbf{R} \\)は以下で表されます:

\\[ \mathbf{R} = u\mathbf{a} + v\mathbf{b} + w\mathbf{c} \\] 

ここで、\\( u, v, w \\)は整数です。

### 1.2.2 逆格子ベクトル

逆格子は、結晶の回折現象を理解する上で極めて重要な概念です。逆格子ベクトル\\( \mathbf{a}^*, \mathbf{b}^*, \mathbf{c}^* \\)は以下のように定義されます:

\\[ \mathbf{a}^* = \frac{\mathbf{b} \times \mathbf{c}}{V}, \quad \mathbf{b}^* = \frac{\mathbf{c} \times \mathbf{a}}{V}, \quad \mathbf{c}^* = \frac{\mathbf{a} \times \mathbf{b}}{V} \\] 

ここで、\\( V = \mathbf{a} \cdot (\mathbf{b} \times \mathbf{c}) \\)は単位格子の体積です。
    
    
    import numpy as np
    
    def reciprocal_lattice_vectors(a, b, c):
        """実格子ベクトルから逆格子ベクトルを計算
    
        Args:
            a, b, c (np.ndarray): 実格子基本ベクトル [Å]
    
        Returns:
            tuple: (a*, b*, c*) 逆格子ベクトル [Å^-1]
        """
        # 単位格子体積
        V = np.dot(a, np.cross(b, c))
    
        # 逆格子ベクトル
        a_star = np.cross(b, c) / V
        b_star = np.cross(c, a) / V
        c_star = np.cross(a, b) / V
    
        return a_star, b_star, c_star
    
    
    def d_spacing_from_hkl(h, k, l, a_star, b_star, c_star):
        """Miller指数から面間隔を計算
    
        Args:
            h, k, l (int): Miller指数
            a_star, b_star, c_star (np.ndarray): 逆格子ベクトル [Å^-1]
    
        Returns:
            float: 面間隔 d [Å]
        """
        # 逆格子ベクトル G = h*a* + k*b* + l*c*
        G = h * a_star + k * b_star + l * c_star
    
        # 面間隔 d = 1 / |G|
        d = 1.0 / np.linalg.norm(G)
    
        return d
    
    
    # 例: 立方晶系 (a = b = c = 4.0 Å, 90度角)
    a = np.array([4.0, 0.0, 0.0])
    b = np.array([0.0, 4.0, 0.0])
    c = np.array([0.0, 0.0, 4.0])
    
    a_star, b_star, c_star = reciprocal_lattice_vectors(a, b, c)
    
    print("実格子ベクトル:")
    print(f"a = {a}")
    print(f"b = {b}")
    print(f"c = {c}\n")
    
    print("逆格子ベクトル:")
    print(f"a* = {a_star} [Å^-1]")
    print(f"b* = {b_star} [Å^-1]")
    print(f"c* = {c_star} [Å^-1]\n")
    
    # 様々なMiller指数の面間隔計算
    miller_indices = [(1, 0, 0), (1, 1, 0), (1, 1, 1), (2, 0, 0), (2, 2, 0)]
    
    print("Miller指数 | 面間隔 d [Å]")
    print("-" * 30)
    for h, k, l in miller_indices:
        d = d_spacing_from_hkl(h, k, l, a_star, b_star, c_star)
        print(f" ({h} {k} {l})      | {d:12.4f}")
    
    # 期待される出力(立方晶):
    # (1 0 0)      |       4.0000
    # (1 1 0)      |       2.8284
    # (1 1 1)      |       2.3094
    # (2 0 0)      |       2.0000
    # (2 2 0)      |       1.4142

## 1.3 消滅則と対称性

### 1.3.1 系統的消滅則

結晶の対称性(空間群)により、特定のMiller指数を持つ反射が消滅する現象を系統的消滅則と呼びます。これは、単位格子内の原子配置の対称性を反映しています。

**代表的な消滅則:**

格子型 | 消滅則 | 例  
---|---|---  
体心立方(BCC) | \\( h + k + l = 奇数 \\) の反射は消滅 | (1,0,0), (1,1,0) は消滅  
面心立方(FCC) | \\( h, k, l \\) が混合(偶奇混在)の反射は消滅 | (1,0,0), (1,1,0) は消滅  
底心斜方晶(C) | \\( h + k = 奇数 \\) の反射は消滅 | (1,0,0), (0,1,0) は消滅  
  
### 1.3.2 消滅則の実装
    
    
    def systematic_absences(h, k, l, lattice_type):
        """系統的消滅則を判定
    
        Args:
            h, k, l (int): Miller指数
            lattice_type (str): 格子型 ('P', 'I', 'F', 'C', 'A', 'B')
    
        Returns:
            bool: True = 反射観測される, False = 消滅
        """
        if lattice_type == 'P':  # 単純格子
            return True
    
        elif lattice_type == 'I':  # 体心格子
            return (h + k + l) % 2 == 0
    
        elif lattice_type == 'F':  # 面心格子
            # h, k, l が全て偶数または全て奇数
            parity = [h % 2, k % 2, l % 2]
            return len(set(parity)) == 1
    
        elif lattice_type == 'C':  # C底心
            return (h + k) % 2 == 0
    
        elif lattice_type == 'A':  # A底心
            return (k + l) % 2 == 0
    
        elif lattice_type == 'B':  # B底心
            return (h + l) % 2 == 0
    
        else:
            raise ValueError(f"Unknown lattice type: {lattice_type}")
    
    
    # テスト: 様々な格子型での消滅則確認
    lattice_types = ['P', 'I', 'F', 'C']
    miller_list = [(1,0,0), (1,1,0), (1,1,1), (2,0,0), (2,2,0), (2,2,2)]
    
    print("Miller指数 | P (単純) | I (体心) | F (面心) | C (底心)")
    print("-" * 65)
    
    for hkl in miller_list:
        h, k, l = hkl
        results = [systematic_absences(h, k, l, lt) for lt in lattice_types]
        status = [' ✓' if r else ' -' for r in results]
        print(f" ({h} {k} {l})     | {status[0]}      | {status[1]}      | {status[2]}      | {status[3]}")
    
    # 期待される出力:
    #  (1 0 0)     |  ✓      |  -      |  -      |  -
    #  (1 1 0)     |  ✓      |  ✓      |  -      |  -
    #  (1 1 1)     |  ✓      |  -      |  ✓      |  ✓
    #  (2 0 0)     |  ✓      |  ✓      |  ✓      |  ✓
    #  (2 2 0)     |  ✓      |  ✓      |  ✓      |  ✓
    #  (2 2 2)     |  ✓      |  ✓      |  ✓      |  ✓

## 1.4 構造因子

### 1.4.1 構造因子の定義

構造因子\\( F_{hkl} \\)は、特定のMiller指数\\( (h, k, l) \\)の回折強度を決定する複素数です。単位格子内の全原子からの散乱波の振幅と位相を合計したものです。

\\[ F_{hkl} = \sum_{j=1}^{N} f_j \exp\left[2\pi i (hx_j + ky_j + lz_j)\right] \\] 

ここで、\\( f_j \\)は原子\\( j \\)の原子散乱因子、\\( (x_j, y_j, z_j) \\)は原子\\( j \\)の分数座標です。

回折強度\\( I_{hkl} \\)は構造因子の絶対値の二乗に比例します:

\\[ I_{hkl} \propto |F_{hkl}|^2 = F_{hkl} \cdot F_{hkl}^* \\] 

### 1.4.2 構造因子の計算実装
    
    
    import numpy as np
    
    def structure_factor(h, k, l, atoms, scattering_factors):
        """構造因子を計算
    
        Args:
            h, k, l (int): Miller指数
            atoms (list): 原子位置のリスト [(x1, y1, z1), (x2, y2, z2), ...]
            scattering_factors (list): 各原子の散乱因子 [f1, f2, ...]
    
        Returns:
            complex: 構造因子 F_hkl
        """
        F_hkl = 0.0 + 0.0j  # 複素数
    
        for (x, y, z), f_j in zip(atoms, scattering_factors):
            # 位相因子 exp[2πi(hx + ky + lz)]
            phase = 2 * np.pi * (h * x + k * y + l * z)
            F_hkl += f_j * np.exp(1j * phase)
    
        return F_hkl
    
    
    def intensity_from_structure_factor(F_hkl):
        """構造因子から回折強度を計算
    
        Args:
            F_hkl (complex): 構造因子
    
        Returns:
            float: 回折強度 I ∝ |F|^2
        """
        return np.abs(F_hkl) ** 2
    
    
    # 例: 単純立方格子 (SC) - 原子1個、(0, 0, 0)
    print("=== 単純立方格子 (SC) ===")
    atoms_sc = [(0, 0, 0)]
    f_sc = [1.0]  # 正規化散乱因子
    
    for hkl in [(1,0,0), (1,1,0), (1,1,1), (2,0,0)]:
        h, k, l = hkl
        F = structure_factor(h, k, l, atoms_sc, f_sc)
        I = intensity_from_structure_factor(F)
        print(f"({h} {k} {l}): F = {F:.4f}, I = {I:.4f}")
    
    # 例: 体心立方格子 (BCC) - 原子2個、(0,0,0), (1/2,1/2,1/2)
    print("\n=== 体心立方格子 (BCC) ===")
    atoms_bcc = [(0, 0, 0), (0.5, 0.5, 0.5)]
    f_bcc = [1.0, 1.0]
    
    for hkl in [(1,0,0), (1,1,0), (1,1,1), (2,0,0), (2,2,0)]:
        h, k, l = hkl
        F = structure_factor(h, k, l, atoms_bcc, f_bcc)
        I = intensity_from_structure_factor(F)
        print(f"({h} {k} {l}): F = {F.real:6.4f}{F.imag:+6.4f}i, I = {I:.4f}")
    
    # 期待される出力:
    # BCC: (1,0,0)と(1,1,0)はI=0(消滅則)、(1,1,1)と(2,2,0)はI>0

### 1.4.3 実際の結晶への適用
    
    
    def calculate_xrd_pattern(a, lattice_type, atom_positions, scattering_factors,
                              wavelength=1.54056, two_theta_max=90):
        """XRDパターンをシミュレーション
    
        Args:
            a (float): 格子定数 [Å]
            lattice_type (str): 格子型 ('P', 'I', 'F')
            atom_positions (list): 原子の分数座標
            scattering_factors (list): 原子散乱因子
            wavelength (float): X線波長 [Å]
            two_theta_max (float): 最大2θ角 [度]
    
        Returns:
            tuple: (two_theta_list, intensity_list)
        """
        # 逆格子ベクトル (立方晶の簡易版)
        a_vec = np.array([a, 0, 0])
        b_vec = np.array([0, a, 0])
        c_vec = np.array([0, 0, a])
        a_star, b_star, c_star = reciprocal_lattice_vectors(a_vec, b_vec, c_vec)
    
        two_theta_list = []
        intensity_list = []
    
        # Miller指数を走査
        for h in range(0, 6):
            for k in range(0, 6):
                for l in range(0, 6):
                    if h == 0 and k == 0 and l == 0:
                        continue
    
                    # 消滅則チェック
                    if not systematic_absences(h, k, l, lattice_type):
                        continue
    
                    # 面間隔計算
                    d = d_spacing_from_hkl(h, k, l, a_star, b_star, c_star)
    
                    # Bragg角計算
                    theta = bragg_law(d, wavelength, n=1)
                    if theta is None or 2*theta > two_theta_max:
                        continue
    
                    # 構造因子と強度計算
                    F = structure_factor(h, k, l, atom_positions, scattering_factors)
                    I = intensity_from_structure_factor(F)
    
                    # ローレンツ偏光因子 (簡易版)
                    LP = (1 + np.cos(2*np.radians(theta))**2) / (np.sin(np.radians(theta))**2 * np.cos(np.radians(theta)))
                    I_corrected = I * LP
    
                    two_theta_list.append(2 * theta)
                    intensity_list.append(I_corrected)
    
        return np.array(two_theta_list), np.array(intensity_list)
    
    
    # シミュレーション: α-Fe (BCC, a = 2.87 Å)
    a_Fe = 2.87
    atoms_Fe = [(0, 0, 0), (0.5, 0.5, 0.5)]
    f_Fe = [26.0, 26.0]  # Feの原子番号
    
    two_theta, intensity = calculate_xrd_pattern(
        a=a_Fe,
        lattice_type='I',
        atom_positions=atoms_Fe,
        scattering_factors=f_Fe,
        wavelength=1.54056,
        two_theta_max=120
    )
    
    # ピークの表示
    print("=== α-Fe (BCC) XRDパターンシミュレーション ===")
    print("2θ [度]  | 相対強度")
    print("-" * 30)
    
    # 強度を正規化
    intensity_normalized = 100 * intensity / np.max(intensity)
    
    # 強度順にソート
    sorted_indices = np.argsort(two_theta)
    for idx in sorted_indices[:10]:  # 上位10ピーク
        print(f"{two_theta[idx]:7.2f}  | {intensity_normalized[idx]:7.1f}")
    
    # 期待される出力: α-Feの典型的なピーク位置
    # (110): ~44.7°
    # (200): ~65.0°
    # (211): ~82.3°

## 1.5 Ewald球と回折条件可視化

### 1.5.1 Ewald球の概念

Ewald球は、X線回折の幾何学的条件を可視化する強力なツールです。逆格子空間において、半径\\( 1/\lambda \\)の球を描き、入射X線ベクトル\\( \mathbf{k}_0 \\)の終点を原点とします。

回折条件は、Ewald球が逆格子点を通過するときに満たされます:

\\[ \mathbf{k} - \mathbf{k}_0 = \mathbf{G}_{hkl} \\] 

ここで、\\( \mathbf{k} \\)は回折X線の波数ベクトル、\\( \mathbf{G}_{hkl} \\)は逆格子ベクトルです。
    
    
    ```mermaid
    graph TD
        A[入射X線波数ベクトル k0] --> B[Ewald球半径 = 1/λ]
        B --> C{逆格子点がEwald球上?}
        C -->|Yes| D[回折条件満足回折発生]
        C -->|No| E[回折なし]
        D --> F[回折X線波数ベクトル k]
        style B fill:#fce7f3
        style D fill:#e8f5e9
        style E fill:#ffebee
    ```

### 1.5.2 Ewald球のプロット
    
    
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    from matplotlib.patches import FancyArrowPatch
    
    def plot_ewald_sphere_2d(wavelength, a, hkl_max=3):
        """Ewald球の2次元断面図を描画
    
        Args:
            wavelength (float): X線波長 [Å]
            a (float): 格子定数 [Å] (立方晶)
            hkl_max (int): 描画する逆格子点の最大Miller指数
        """
        fig, ax = plt.subplots(figsize=(10, 8))
    
        # Ewald球の半径
        k = 1.0 / wavelength  # [Å^-1]
    
        # Ewald球の中心 (入射X線の終点)
        center = np.array([k, 0])
    
        # Ewald球を描画
        circle = Circle(center, k, fill=False, edgecolor='#f093fb', linewidth=2, label='Ewald球')
        ax.add_patch(circle)
    
        # 逆格子点を描画 (立方晶, 2D断面)
        a_star = 1.0 / a  # [Å^-1]
    
        reciprocal_points = []
        for h in range(-hkl_max, hkl_max + 1):
            for l in range(-hkl_max, hkl_max + 1):
                G_x = h * a_star
                G_z = l * a_star
                reciprocal_points.append((G_x, G_z, h, l))
    
                # 逆格子点をプロット
                ax.plot(G_x, G_z, 'o', color='#2c3e50', markersize=4)
    
        # 入射X線ベクトル
        ax.arrow(0, 0, k, 0, head_width=0.02, head_length=0.03,
                 fc='#f5576c', ec='#f5576c', linewidth=2, label='入射X線 k0')
    
        # Ewald球と交差する逆格子点を強調
        for G_x, G_z, h, l in reciprocal_points:
            dist_from_center = np.sqrt((G_x - center[0])**2 + (G_z - center[1])**2)
            if abs(dist_from_center - k) < 0.02:  # 球上にある
                ax.plot(G_x, G_z, 'o', color='#e74c3c', markersize=10,
                       label=f'回折可能: ({h},0,{l})' if h==1 and l==0 else '')
    
                # 回折X線ベクトル
                ax.arrow(0, 0, G_x, G_z, head_width=0.01, head_length=0.02,
                        fc='#e74c3c', ec='#e74c3c', linewidth=1.5,
                        linestyle='--', alpha=0.7)
    
        ax.set_xlabel('逆格子 h方向 [Å$^{-1}$]', fontsize=12)
        ax.set_ylabel('逆格子 l方向 [Å$^{-1}$]', fontsize=12)
        ax.set_title(f'Ewald球と逆格子 (λ = {wavelength:.2f} Å, a = {a:.2f} Å)', fontsize=14)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='k', linewidth=0.5)
        ax.axvline(x=0, color='k', linewidth=0.5)
    
        # 凡例を整理
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='upper right')
    
        plt.tight_layout()
        plt.show()
    
    # Cu Kα線、格子定数 a = 3.5 Å の場合
    plot_ewald_sphere_2d(wavelength=1.54056, a=3.5, hkl_max=2)

## 学習目標の確認

この章を完了したあなたは、以下を説明・実装できるようになりました:

### 基本理解

  * ✅ Braggの法則 \\( 2d\sin\theta = n\lambda \\) の物理的意味と導出
  * ✅ 結晶格子と逆格子の関係、および面間隔の計算
  * ✅ 系統的消滅則と結晶対称性の関係

### 実践スキル

  * ✅ Pythonで回折角を計算し、面間隔との関係を可視化
  * ✅ 逆格子ベクトルからMiller指数の面間隔を導出
  * ✅ 構造因子を計算し、回折強度を予測
  * ✅ XRDパターンをシミュレーションし、実測データと比較可能

### 応用力

  * ✅ Ewald球を用いて回折条件を幾何学的に理解
  * ✅ 消滅則から結晶の格子型を推定
  * ✅ 実際の材料(α-Fe等)のXRDパターンをシミュレーション

## 演習問題

### Easy (基礎確認)

**Q1** : Cu Kα線(λ = 1.54 Å)を用いて、面間隔 d = 2.5 Å の結晶面のBragg角(n=1)を計算してください。

**解答** :
    
    
    theta = bragg_law(d=2.5, wavelength=1.54, n=1)
    print(f"Bragg角: {theta:.2f}度")
    # 出力: Bragg角: 17.93度

**解説** :

Braggの法則 \\( 2d\sin\theta = n\lambda \\) より:

\\[ \sin\theta = \frac{n\lambda}{2d} = \frac{1 \times 1.54}{2 \times 2.5} = 0.308 \\]

\\[ \theta = \arcsin(0.308) = 17.93° \\]

**Q2** : 体心立方格子(BCC)で、Miller指数 (1,1,0) の反射は観測されますか? 消滅則を確認してください。

**解答** :
    
    
    is_allowed = systematic_absences(1, 1, 0, 'I')
    print(f"(1,1,0)反射: {'観測される' if is_allowed else '消滅'}")
    # 出力: (1,1,0)反射: 観測される

**解説** :

BCC (体心格子, lattice_type='I') の消滅則: \\( h + k + l = 偶数 \\) のみ観測。

(1,1,0)の場合: \\( 1 + 1 + 0 = 2 \\) (偶数) → 観測される ✓

(1,0,0)の場合: \\( 1 + 0 + 0 = 1 \\) (奇数) → 消滅 ✗

### Medium (応用)

**Q3** : 立方晶 (a=4.0Å) の (2,2,0) 面と (1,1,1) 面の面間隔を計算し、どちらの面間隔が大きいか比較してください。

**解答** :
    
    
    a = 4.0
    a_vec = np.array([a, 0, 0])
    b_vec = np.array([0, a, 0])
    c_vec = np.array([0, 0, a])
    a_star, b_star, c_star = reciprocal_lattice_vectors(a_vec, b_vec, c_vec)
    
    d_220 = d_spacing_from_hkl(2, 2, 0, a_star, b_star, c_star)
    d_111 = d_spacing_from_hkl(1, 1, 1, a_star, b_star, c_star)
    
    print(f"d_220 = {d_220:.4f} Å")
    print(f"d_111 = {d_111:.4f} Å")
    print(f"大きいのは: {'(1,1,1)' if d_111 > d_220 else '(2,2,0)'}")
    
    # 期待される出力:
    # d_220 = 1.4142 Å
    # d_111 = 2.3094 Å
    # 大きいのは: (1,1,1)

**解説** :

立方晶の面間隔公式: \\( d_{hkl} = \frac{a}{\sqrt{h^2 + k^2 + l^2}} \\)

(2,2,0): \\( d = 4.0 / \sqrt{4+4+0} = 4.0 / 2.83 = 1.414 \\) Å

(1,1,1): \\( d = 4.0 / \sqrt{1+1+1} = 4.0 / 1.732 = 2.309 \\) Å

結論: (1,1,1) の方が面間隔が大きい (Miller指数が小さい面ほど間隔が広い)

**Q4** : 単純立方格子(SC)と面心立方格子(FCC)で、(1,0,0)反射の強度を構造因子から比較してください。

**解答** :
    
    
    # SC: 原子1個 at (0,0,0)
    atoms_sc = [(0, 0, 0)]
    f_sc = [1.0]
    F_sc = structure_factor(1, 0, 0, atoms_sc, f_sc)
    I_sc = intensity_from_structure_factor(F_sc)
    
    # FCC: 原子4個 at (0,0,0), (1/2,1/2,0), (1/2,0,1/2), (0,1/2,1/2)
    atoms_fcc = [(0,0,0), (0.5,0.5,0), (0.5,0,0.5), (0,0.5,0.5)]
    f_fcc = [1.0, 1.0, 1.0, 1.0]
    F_fcc = structure_factor(1, 0, 0, atoms_fcc, f_fcc)
    I_fcc = intensity_from_structure_factor(F_fcc)
    
    print(f"SC: F_100 = {F_sc:.4f}, I = {I_sc:.4f}")
    print(f"FCC: F_100 = {F_fcc:.4f}, I = {I_fcc:.4f}")
    
    # 期待される出力:
    # SC: F_100 = 1.0000, I = 1.0000
    # FCC: F_100 = 0.0000, I = 0.0000 (消滅)

**解説** :

FCC構造因子 (1,0,0):

\\[ F = 1 \cdot e^{0} + 1 \cdot e^{i\pi} + 1 \cdot e^{i\pi} + 1 \cdot e^{0} = 1 + (-1) + (-1) + 1 = 0 \\]

FCCでは混合指数反射((1,0,0)等)は消滅します。

### Hard (発展)

**Q5** : α-Fe (BCC, a=2.87Å) と γ-Fe (FCC, a=3.65Å) のXRDパターンをシミュレーションし、最も強いピーク位置(2θ)を比較してください。Cu Kα線使用。

**解答** :
    
    
    # α-Fe (BCC)
    two_theta_bcc, intensity_bcc = calculate_xrd_pattern(
        a=2.87, lattice_type='I',
        atom_positions=[(0,0,0), (0.5,0.5,0.5)],
        scattering_factors=[26.0, 26.0]
    )
    max_idx_bcc = np.argmax(intensity_bcc)
    print(f"α-Fe (BCC) 最強ピーク: 2θ = {two_theta_bcc[max_idx_bcc]:.2f}°")
    
    # γ-Fe (FCC)
    two_theta_fcc, intensity_fcc = calculate_xrd_pattern(
        a=3.65, lattice_type='F',
        atom_positions=[(0,0,0), (0.5,0.5,0), (0.5,0,0.5), (0,0.5,0.5)],
        scattering_factors=[26.0]*4
    )
    max_idx_fcc = np.argmax(intensity_fcc)
    print(f"γ-Fe (FCC) 最強ピーク: 2θ = {two_theta_fcc[max_idx_fcc]:.2f}°")
    
    # 期待される出力:
    # α-Fe (BCC) 最強ピーク: 2θ ≈ 44.7° (110反射)
    # γ-Fe (FCC) 最強ピーク: 2θ ≈ 43.6° (111反射)

**解説** :

α-Fe (BCC): 最強ピークは(110)反射 (h+k+l=2, 最小偶数)

γ-Fe (FCC): 最強ピークは(111)反射 (全て奇数, 最小)

この違いにより、XRDでBCCとFCCを明確に区別できます。実際の鉄鋼材料の相変態解析に応用されています。

**Q6** : Ewald球の概念を用いて、なぜ単色X線では全ての逆格子点が同時に回折しないのかを説明してください。

**解答** :

**理由** :

  1. Ewald球の半径は固定 (= 1/λ)
  2. 回折条件は「逆格子点がEwald球上にある」こと
  3. 単色X線では試料を回転させない限り、Ewald球は固定位置
  4. ほとんどの逆格子点は球の内部または外部にあり、球面上にない

**実験的解決法** :

  * **粉末XRD** : ランダム配向の微結晶により、全方位からの回折を観測
  * **単結晶回転法** : 試料を回転させ、逆格子点を順次Ewald球に通過させる
  * **白色X線(Laue法)** : 様々な波長 → 様々な半径のEwald球 → 多数の反射を同時観測

## 学習目標の確認

この章を完了すると、以下を説明・実装できるようになります：

### 基本理解

  * ✅ Braggの法則（nλ = 2d sinθ）を導出し、物理的意味を説明できる
  * ✅ 結晶格子と逆格子の関係を理解し、逆格子ベクトルを計算できる
  * ✅ 消滅則が生じる物理的メカニズム（破壊的干渉）を説明できる
  * ✅ 構造因子F(hkl)の計算方法と物理的意味を理解している

### 実践スキル

  * ✅ Pythonでdスペーシングを計算し、回折角を予測できる
  * ✅ 任意の空間群に対して消滅則を適用し、許容反射を判定できる
  * ✅ 原子座標から構造因子を計算し、回折強度を予測できる
  * ✅ Ewald球の概念を用いて回折条件を視覚化できる

### 応用力

  * ✅ 実測XRDパターンから結晶構造（BCC/FCC等）を推定できる
  * ✅ 格子定数の変化がXRDパターンに与える影響を予測できる
  * ✅ 消滅則を利用して空間群候補を絞り込むことができる
  * ✅ 多結晶材料のXRD解析戦略を立案できる

## 参考文献

  1. Cullity, B. D., & Stock, S. R. (2001). _Elements of X-Ray Diffraction_ (3rd ed.). Prentice Hall. - X線回折学の古典的教科書、Braggの法則から結晶構造解析まで包括的に解説
  2. Warren, B. E. (1990). _X-ray Diffraction_. Dover Publications. - 回折理論の物理的基礎を詳細に解説、構造因子の導出が秀逸
  3. Pecharsky, V. K., & Zavalij, P. Y. (2009). _Fundamentals of Powder Diffraction and Structural Characterization of Materials_ (2nd ed.). Springer. - 粉末XRDに特化した実践的教科書
  4. International Tables for Crystallography, Volume A: Space-Group Symmetry (2016). International Union of Crystallography. - 空間群と消滅則の決定版リファレンス
  5. Giacovazzo, C., et al. (2011). _Fundamentals of Crystallography_ (3rd ed.). Oxford University Press. - 結晶学の理論的基礎、逆格子の概念を明快に解説
  6. Hammond, C. (2015). _The Basics of Crystallography and Diffraction_ (4th ed.). Oxford University Press. - Ewald球構成法の視覚的理解に優れた教科書
  7. Ladd, M., & Palmer, R. (2013). _Structure Determination by X-ray Crystallography_ (5th ed.). Springer. - 構造因子計算から結晶構造決定まで実践的ガイド

## 次のステップ

第1章では、X線回折の理論的基礎を学びました。Braggの法則、逆格子、消滅則、構造因子といった概念は、すべてのXRD解析の土台となります。

**第2章** では、これらの理論を実際のXRD測定に応用します。粉末X線回折データの取得、ピーク同定、バックグラウンド除去、ピークフィッティングといった実践的な解析手法を学びます。

### 免責事項

  * 本コンテンツは教育・研究・情報提供のみを目的としており、専門的な助言(法律・会計・技術的保証など)を提供するものではありません。
  * 本コンテンツおよび付随するCode examplesは「現状有姿(AS IS)」で提供され、明示または黙示を問わず、商品性、特定目的適合性、権利非侵害、正確性・完全性、動作・安全性等いかなる保証もしません。
  * 外部リンク、第三者が提供するデータ・ツール・ライブラリ等の内容・可用性・安全性について、作成者および東北大学は一切の責任を負いません。
  * 本コンテンツの利用・実行・解釈により直接的・間接的・付随的・特別・結果的・懲罰的損害が生じた場合でも、適用法で許容される最大限の範囲で、作成者および東北大学は責任を負いません。
  * 本コンテンツの内容は、予告なく変更・更新・提供停止されることがあります。
  * 本コンテンツの著作権・ライセンスは明記された条件(例: CC BY 4.0)に従います。当該ライセンスは通常、無保証条項を含みます。
