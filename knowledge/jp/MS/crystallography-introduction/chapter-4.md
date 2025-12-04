---
title: 第4章：X線回折の原理と応用
chapter_title: 第4章：X線回折の原理と応用
subtitle: ブラッグの法則から実測データ解析まで
reading_time: 30分
code_examples: 8
---

🌐 JP | [🇬🇧 EN](<../../../en/MS/crystallography-introduction/chapter-4.html>) | Last sync: 2025-11-16

[AI寺子屋トップ](<../../index.html>) > [MS Dojo](<../index.html>) > [結晶学入門](<index.html>) > 第4章 

## 学習目標

この章を学ぶことで、以下の知識とスキルを習得できます：

  * **X線と物質の相互作用** の基礎を理解する
  * **ブラッグの法則 nλ = 2d sinθ** を導出し、物理的意味を説明できる
  * **構造因子 F hkl**を計算し、回折強度への影響を理解する
  * **消滅則（系統的消滅）** から結晶構造を推定できる
  * **粉末X線回折パターン** を解釈し、ピーク同定ができる
  * **リートベルト解析** の基礎概念を理解する
  * **実測XRDデータ** をPythonで解析できる

## 1\. X線と結晶の相互作用

### 1.1 X線とは何か

**X線（X-ray）** は、波長が約0.01〜10 nm（10 Å）の電磁波です。 材料科学では、主に**Cu Kα線（λ = 1.5406 Å）** や**Mo Kα線（λ = 0.7107 Å）** が使用されます。 

X線の波長が原子間距離（数Å）と同程度であるため、結晶格子による**回折現象** が起こります。

#### 主要なX線源

X線源 | 波長 (Å) | エネルギー (keV) | 主な用途  
---|---|---|---  
Cu Kα | 1.5406 | 8.05 | 粉末XRD、一般的な構造解析  
Mo Kα | 0.7107 | 17.48 | 単結晶XRD、短波長が必要な場合  
Co Kα | 1.7902 | 6.93 | 鉄を含む試料（蛍光X線回避）  
シンクロトロン | 可変 | 可変 | 高輝度、高分解能測定  
  
### 1.2 X線回折の基本原理

X線が結晶に入射すると、以下のプロセスが起こります：
    
    
    ```mermaid
    flowchart TD
                    A[X線入射] --> B[各原子による散乱]
                    B --> C{散乱波の干渉}
                    C -->|位相が揃う| D[強め合い：回折ピーク]
                    C -->|位相がずれる| E[弱め合い：消滅]
                    D --> F[検出器でピーク観測]
                    E --> G[バックグラウンド]
    
                    style A fill:#e3f2fd
                    style B fill:#e3f2fd
                    style C fill:#fff3e0
                    style D fill:#e8f5e9
                    style E fill:#ffebee
                    style F fill:#e8f5e9
                    style G fill:#ffebee
    ```

結晶内の各原子がX線を散乱し、散乱波が**干渉** します。 特定の角度で散乱波の位相が揃うと**強め合い** 、**回折ピーク** として観測されます。 

## 2\. ブラッグの法則

### 2.1 ブラッグの法則の導出

1913年、ウィリアム・ローレンス・ブラッグ（W. L. Bragg）とその父ウィリアム・ヘンリー・ブラッグ（W. H. Bragg）は、 X線回折を**結晶面による鏡面反射** として扱う単純で強力なモデルを提案しました。 

#### ブラッグの法則（Bragg's Law）

$$ n\lambda = 2d_{hkl}\sin\theta $$ 

ここで：

  * **n** ：反射の次数（通常は1）
  * **λ** ：X線の波長
  * **d hkl**：(hkl)面の面間隔
  * **θ** ：ブラッグ角（入射角 = 反射角）

#### 導出の考え方

面間隔dで平行に並ぶ結晶面に角度θでX線が入射する場合を考えます。 上の面で反射した波と、下の面で反射した波の**光路差** が波長λの整数倍であれば、 強め合いが起こります。 

光路差は幾何学的に以下のように計算されます：

$$ \text{光路差} = 2d\sin\theta $$ 

強め合いの条件は：

$$ 2d\sin\theta = n\lambda \quad (n = 1, 2, 3, \ldots) $$ 

#### 重要な注意点

ブラッグの法則は**回折が起こる必要条件** であり、**十分条件ではありません** 。 実際には、**構造因子 F hkl**がゼロでないことも必要です（後述）。 

### 2.2 ブラッグの法則を使った計算

#### コード例1：ブラッグ角の計算

シリコン（Si）の主要な結晶面からの回折角を計算します：
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    def cubic_d_spacing(a, h, k, l):
        """
        立方晶の面間隔を計算
    
        Parameters:
        -----------
        a : float
            格子定数 (Å)
        h, k, l : int
            ミラー指数
    
        Returns:
        --------
        float : 面間隔 (Å)
        """
        return a / np.sqrt(h**2 + k**2 + l**2)
    
    def bragg_angle(d_hkl, wavelength, n=1):
        """
        ブラッグの法則から回折角 2θ を計算
    
        Parameters:
        -----------
        d_hkl : float
            面間隔 (Å)
        wavelength : float
            X線波長 (Å)
        n : int
            反射の次数
    
        Returns:
        --------
        float : 回折角 2θ (度)、またはNone（回折不可の場合）
        """
        sin_theta = n * wavelength / (2 * d_hkl)
        if abs(sin_theta) > 1:
            return None  # 回折条件を満たさない
        theta = np.arcsin(sin_theta)
        return np.degrees(2 * theta)  # 2θ を返す
    
    # シリコン（Si）のパラメータ
    a_Si = 5.4310  # Å
    wavelength_CuKa = 1.5406  # Å
    
    print("=== シリコン（Si）のX線回折パターン予測 ===")
    print(f"格子定数: a = {a_Si} Å")
    print(f"X線波長: λ = {wavelength_CuKa} Å (Cu Kα)\n")
    
    # 主要な結晶面
    planes = [
        (1, 1, 1), (2, 2, 0), (3, 1, 1),
        (4, 0, 0), (3, 3, 1), (4, 2, 2)
    ]
    
    print(f"{'(hkl)':<10} {'d (Å)':<12} {'2θ (度)':<12}")
    print("-" * 40)
    
    results = []
    for hkl in planes:
        h, k, l = hkl
        d = cubic_d_spacing(a_Si, h, k, l)
        two_theta = bragg_angle(d, wavelength_CuKa)
    
        if two_theta is not None:
            print(f"({h}{k}{l}){'':<8} {d:8.4f}    {two_theta:8.3f}")
            results.append((hkl, two_theta))
    
    # グラフでピークパターンを可視化
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for (h, k, l), two_theta in results:
        ax.axvline(two_theta, color='red', linewidth=2, alpha=0.7)
        ax.text(two_theta, 1.05, f'({h}{k}{l})',
                rotation=90, va='bottom', ha='right', fontsize=9)
    
    ax.set_xlim(20, 100)
    ax.set_ylim(0, 1.2)
    ax.set_xlabel('2θ (度)', fontsize=14, fontweight='bold')
    ax.set_ylabel('相対強度（任意単位）', fontsize=14, fontweight='bold')
    ax.set_title('シリコン（Si）の理論XRDパターン', fontsize=16, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('si_xrd_pattern.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("\nXRDパターンを保存しました: si_xrd_pattern.png")
    

### 2.3 波長と回折角の関係

#### コード例2：異なるX線源での回折パターン比較

Cu Kα線とMo Kα線で同じ結晶面からの回折角がどう変わるかを比較します：
    
    
    def compare_xray_sources():
        """異なるX線源での回折パターンを比較"""
    
        # X線源のパラメータ
        sources = {
            'Cu Kα': 1.5406,
            'Mo Kα': 0.7107,
            'Co Kα': 1.7902
        }
    
        # アルミニウム（Al）のパラメータ
        a_Al = 4.0495  # Å
        planes = [(1, 1, 1), (2, 0, 0), (2, 2, 0), (3, 1, 1)]
    
        print("=== 異なるX線源での回折角比較（Al） ===\n")
        print(f"格子定数: a = {a_Al} Å\n")
    
        # 各X線源での計算
        fig, ax = plt.subplots(figsize=(14, 8))
        colors = {'Cu Kα': 'red', 'Mo Kα': 'blue', 'Co Kα': 'green'}
    
        for source_name, wavelength in sources.items():
            print(f"--- {source_name} (λ = {wavelength} Å) ---")
            print(f"{'(hkl)':<10} {'d (Å)':<12} {'2θ (度)':<12}")
            print("-" * 40)
    
            y_offset = list(sources.keys()).index(source_name) * 0.3
    
            for hkl in planes:
                h, k, l = hkl
                d = cubic_d_spacing(a_Al, h, k, l)
                two_theta = bragg_angle(d, wavelength)
    
                if two_theta is not None:
                    print(f"({h}{k}{l}){'':<8} {d:8.4f}    {two_theta:8.3f}")
    
                    # 棒グラフとして表示
                    ax.plot([two_theta, two_theta], [y_offset, y_offset + 0.25],
                           color=colors[source_name], linewidth=3)
                    if y_offset == 0:  # 最初のソースのみラベル表示
                        ax.text(two_theta, y_offset + 0.28, f'({h}{k}{l})',
                               rotation=90, va='bottom', ha='center', fontsize=8)
            print()
    
        # グラフの装飾
        ax.set_xlim(0, 150)
        ax.set_ylim(-0.1, 1.0)
        ax.set_xlabel('2θ (度)', fontsize=14, fontweight='bold')
        ax.set_yticks([0.125, 0.425, 0.725])
        ax.set_yticklabels(['Cu Kα', 'Mo Kα', 'Co Kα'])
        ax.set_title('異なるX線源によるアルミニウム（Al）の回折パターン比較',
                    fontsize=16, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
    
        # 凡例
        from matplotlib.lines import Line2D
        legend_elements = [Line2D([0], [0], color=color, lw=3, label=name)
                          for name, color in colors.items()]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=12)
    
        plt.tight_layout()
        plt.savefig('xray_source_comparison.png', dpi=150, bbox_inches='tight')
        plt.show()
        print("比較グラフを保存しました: xray_source_comparison.png")
    
    # 実行
    compare_xray_sources()
    

#### 波長選択の実践的考慮

  * **短波長（Mo Kα）** ：高角度まで測定可能、吸収が少ない、単結晶解析に有利
  * **長波長（Cu Kα）** ：低角度での分解能が高い、最も一般的
  * **試料依存** ：鉄を含む試料ではCo Kαを使用（蛍光X線回避）

## 3\. 構造因子と回折強度

### 3.1 構造因子 Fhkl とは

ブラッグの法則を満たしても、全ての反射が観測されるわけではありません。 **構造因子（structure factor）F hkl**が、 実際の回折強度を決定します。 

$$ F_{hkl} = \sum_{j=1}^{N} f_j \exp\left[2\pi i(hx_j + ky_j + lz_j)\right] $$ 

ここで：

  * **f j**：j番目の原子の**原子散乱因子** （原子の電子数に比例）
  * **(x j, yj, zj)**：j番目の原子の**分数座標**
  * **N** ：単位格子内の原子数

**重要** ：Fhkl = 0 の場合、ブラッグの法則を満たしても回折は起こりません。 これが**消滅則（systematic absence）** です。 

### 3.2 単純な構造の構造因子計算

#### コード例3：単純立方・体心立方・面心立方の構造因子
    
    
    import numpy as np
    import pandas as pd
    
    def structure_factor(positions, f_atoms, h, k, l):
        """
        構造因子 F_hkl を計算
    
        Parameters:
        -----------
        positions : list of tuples
            単位格子内の原子の分数座標 [(x1,y1,z1), (x2,y2,z2), ...]
        f_atoms : list of float
            各原子の原子散乱因子
        h, k, l : int
            ミラー指数
    
        Returns:
        --------
        complex : 構造因子 F_hkl
        """
        F = 0 + 0j
        for (x, y, z), f in zip(positions, f_atoms):
            phase = 2 * np.pi * (h*x + k*y + l*z)
            F += f * np.exp(1j * phase)
        return F
    
    def analyze_structure_factors():
        """異なる格子タイプの構造因子を解析"""
    
        # 格子タイプごとの原子位置
        structures = {
            'SC（単純立方）': [
                (0, 0, 0)
            ],
            'BCC（体心立方）': [
                (0, 0, 0),
                (0.5, 0.5, 0.5)
            ],
            'FCC（面心立方）': [
                (0, 0, 0),
                (0.5, 0.5, 0),
                (0.5, 0, 0.5),
                (0, 0.5, 0.5)
            ],
            'Diamond（ダイヤモンド構造）': [
                (0, 0, 0),
                (0.5, 0.5, 0),
                (0.5, 0, 0.5),
                (0, 0.5, 0.5),
                (0.25, 0.25, 0.25),
                (0.75, 0.75, 0.25),
                (0.75, 0.25, 0.75),
                (0.25, 0.75, 0.75)
            ]
        }
    
        # ミラー指数のリスト
        planes = [
            (1, 0, 0), (1, 1, 0), (1, 1, 1),
            (2, 0, 0), (2, 2, 0), (3, 1, 1),
            (2, 2, 2), (4, 0, 0), (3, 3, 1)
        ]
    
        print("=== 異なる格子タイプの構造因子と消滅則 ===\n")
    
        for structure_name, positions in structures.items():
            print(f"\n【{structure_name}】")
            print(f"単位格子内の原子数: {len(positions)}\n")
            print(f"{'(hkl)':<10} {'|F_hkl|^2':<15} {'観測':<10} {'備考'}")
            print("-" * 60)
    
            # すべて同じ原子として原子散乱因子 f = 1 と仮定
            f_atoms = [1.0] * len(positions)
    
            for hkl in planes:
                h, k, l = hkl
                F = structure_factor(positions, f_atoms, h, k, l)
                F_squared = abs(F)**2
    
                # 観測可否の判定（|F|^2 > 0.01を観測可能とする）
                observable = "○" if F_squared > 0.01 else "×"
    
                # 消滅条件の説明
                remarks = ""
                if structure_name == 'BCC（体心立方）':
                    if (h + k + l) % 2 != 0:
                        remarks = "h+k+l が奇数 → 消滅"
                elif structure_name == 'FCC（面心立方）':
                    if not (h % 2 == k % 2 == l % 2):
                        remarks = "h,k,l が混合 → 消滅"
    
                print(f"({h}{k}{l}){'':<8} {F_squared:12.4f}   {observable:<10} {remarks}")
    
        print("\n" + "="*60)
        print("消滅則のまとめ：")
        print("  SC : すべての反射が観測される")
        print("  BCC: h+k+l が偶数のみ観測される")
        print("  FCC: h, k, l がすべて偶数またはすべて奇数のみ観測される")
        print("  Diamond: FCC + さらに h+k+l=4n (n:整数) のみ強い反射")
        print("="*60)
    
    # 実行
    analyze_structure_factors()
    

#### 消滅則の重要性

消滅則は結晶構造を決定する上で極めて重要です。 例えば、(100)ピークが観測されない場合、単純立方（SC）ではなくBCCまたはFCCであることが分かります。 

### 3.3 回折強度に影響する因子

実際の回折強度 Ihkl は、構造因子以外にも多くの因子に依存します：

$$ I_{hkl} \propto |F_{hkl}|^2 \cdot m_{hkl} \cdot L \cdot P \cdot A \cdot \exp(-2M) $$ 

因子 | 名称 | 物理的意味  
---|---|---  
|Fhkl|2 | 構造因子 | 単位格子内の原子配置の効果  
mhkl | 多重度 | 対称的に等価な面の数  
L | ローレンツ因子 | 結晶の幾何学的効果  
P | 偏光因子 | X線の偏光状態の効果  
A | 吸収因子 | 試料によるX線吸収  
exp(-2M) | 温度因子（デバイ・ワラー因子） | 原子の熱振動による散漫散乱  
  
#### コード例4：多重度とローレンツ偏光因子を含む強度計算
    
    
    from itertools import permutations, product
    
    def multiplicity(h, k, l, crystal_system='cubic'):
        """
        多重度（等価な面の数）を計算
    
        Parameters:
        -----------
        h, k, l : int
            ミラー指数
        crystal_system : str
            結晶系
    
        Returns:
        --------
        int : 多重度
        """
        planes = set()
    
        if crystal_system == 'cubic':
            for perm in permutations([abs(h), abs(k), abs(l)]):
                for signs in product([1, -1], repeat=3):
                    plane = tuple(s * p for s, p in zip(signs, perm))
                    if plane != (0, 0, 0):
                        planes.add(plane)
    
        return len(planes)
    
    def lorentz_polarization_factor(two_theta):
        """
        粉末X線回折のローレンツ・偏光因子
    
        Parameters:
        -----------
        two_theta : float
            回折角 2θ（ラジアン）
    
        Returns:
        --------
        float : LP因子
        """
        theta = two_theta / 2
        LP = (1 + np.cos(two_theta)**2) / (np.sin(theta)**2 * np.cos(theta))
        return LP
    
    def calculate_intensity_with_factors():
        """各種因子を考慮した回折強度の計算"""
    
        # アルミニウム（FCC、a = 4.0495 Å）
        a_Al = 4.0495
        wavelength = 1.5406  # Cu Kα
    
        # FCC構造の原子位置
        fcc_positions = [
            (0, 0, 0),
            (0.5, 0.5, 0),
            (0.5, 0, 0.5),
            (0, 0.5, 0.5)
        ]
        f_Al = 13  # アルミニウムの原子番号（簡易的に原子散乱因子とする）
    
        planes = [
            (1, 1, 1), (2, 0, 0), (2, 2, 0),
            (3, 1, 1), (2, 2, 2), (4, 0, 0)
        ]
    
        print("=== 各種因子を考慮したXRD強度計算（Al, FCC） ===\n")
        print(f"{'(hkl)':<10} {'d (Å)':<10} {'2θ':<10} {'|F|^2':<12} {'m':<6} {'LP':<10} {'I_rel':<10}")
        print("-" * 80)
    
        intensities = []
    
        for hkl in planes:
            h, k, l = hkl
    
            # 面間隔
            d = cubic_d_spacing(a_Al, h, k, l)
    
            # ブラッグ角
            two_theta_deg = bragg_angle(d, wavelength)
            if two_theta_deg is None:
                continue
            two_theta_rad = np.radians(two_theta_deg)
    
            # 構造因子
            F = structure_factor(fcc_positions, [f_Al]*4, h, k, l)
            F_squared = abs(F)**2
    
            # 消滅則チェック（FCCでは混合は消滅）
            if F_squared < 0.01:
                continue
    
            # 多重度
            m = multiplicity(h, k, l, 'cubic')
    
            # ローレンツ偏光因子
            LP = lorentz_polarization_factor(two_theta_rad)
    
            # 相対強度（温度因子・吸収は省略）
            I_rel = F_squared * m * LP
    
            intensities.append((hkl, I_rel))
    
            print(f"({h}{k}{l}){'':<8} {d:8.4f}  {two_theta_deg:8.2f}  {F_squared:10.2f}  {m:<6} {LP:8.4f}  {I_rel:8.2f}")
    
        # 最大強度で規格化
        max_intensity = max(I for _, I in intensities)
    
        print("\n--- 規格化相対強度（最大値 = 100） ---")
        print(f"{'(hkl)':<10} {'相対強度':<15}")
        print("-" * 30)
    
        for hkl, I in intensities:
            I_normalized = 100 * I / max_intensity
            print(f"({hkl[0]}{hkl[1]}{hkl[2]}){'':<8} {I_normalized:8.1f}")
    
    # 実行
    calculate_intensity_with_factors()
    

## 4\. 粉末X線回折パターンの解釈

### 4.1 粉末XRDとは

**粉末X線回折（Powder X-ray Diffraction, PXRD）** は、 微細な結晶粒が**ランダムな向き** で配向した試料に対して行う測定法です。 材料科学で最も頻繁に使用される構造評価手法の一つです。 

#### 粉末XRDの特徴

  * **試料準備が容易** ：単結晶を必要としない
  * **相同定** ：未知試料の結晶相を特定できる
  * **定量分析** ：混合物中の各相の割合を推定可能
  * **格子定数決定** ：ピーク位置から精密な格子定数を求められる
  * **結晶子サイズ** ：ピーク幅からナノ粒子のサイズを推定

### 4.2 XRDパターンのシミュレーション

#### コード例5：完全なXRDパターンシミュレーション

ピーク形状（ガウス関数）とバックグラウンドを含む現実的なXRDパターンを生成します：
    
    
    def gaussian_peak(two_theta, center, intensity, fwhm):
        """
        ガウス型ピーク関数
    
        Parameters:
        -----------
        two_theta : array
            2θ の配列
        center : float
            ピーク中心位置
        intensity : float
            ピーク強度
        fwhm : float
            半値全幅（Full Width at Half Maximum）
    
        Returns:
        --------
        array : ガウスピークの強度
        """
        sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
        return intensity * np.exp(-((two_theta - center)**2) / (2 * sigma**2))
    
    def simulate_xrd_pattern(material_name, a, c=None, crystal_system='cubic',
                            positions=None, f_atoms=None,
                            wavelength=1.5406, two_theta_range=(20, 100),
                            fwhm=0.2, background=50):
        """
        完全なXRDパターンをシミュレーション
    
        Parameters:
        -----------
        material_name : str
            材料名
        a, c : float
            格子定数
        crystal_system : str
            結晶系
        positions : list
            単位格子内の原子の分数座標
        f_atoms : list
            原子散乱因子
        wavelength : float
            X線波長
        two_theta_range : tuple
            2θ の測定範囲
        fwhm : float
            ピークの半値全幅
        background : float
            バックグラウンド強度
    
        Returns:
        --------
        two_theta, intensity : arrays
        """
        # 2θ の配列
        two_theta = np.linspace(two_theta_range[0], two_theta_range[1], 4000)
        intensity = np.ones_like(two_theta) * background  # バックグラウンド
    
        # ピークを計算
        max_hkl = 5
        peaks_info = []
    
        for h in range(max_hkl + 1):
            for k in range(h, max_hkl + 1):
                for l in range(k, max_hkl + 1):
                    if h == 0 and k == 0 and l == 0:
                        continue
    
                    # 面間隔
                    if crystal_system == 'cubic':
                        d = cubic_d_spacing(a, h, k, l)
    
                    # ブラッグ角
                    two_theta_peak = bragg_angle(d, wavelength)
                    if two_theta_peak is None or two_theta_peak > two_theta_range[1]:
                        continue
    
                    # 構造因子
                    if positions is not None and f_atoms is not None:
                        F = structure_factor(positions, f_atoms, h, k, l)
                        F_squared = abs(F)**2
                        if F_squared < 0.01:
                            continue
                    else:
                        F_squared = 1.0
    
                    # 多重度
                    m = multiplicity(h, k, l, crystal_system)
    
                    # ローレンツ偏光因子
                    LP = lorentz_polarization_factor(np.radians(two_theta_peak))
    
                    # ピーク強度
                    I_peak = F_squared * m * LP
    
                    # ガウスピークを追加
                    intensity += gaussian_peak(two_theta, two_theta_peak, I_peak, fwhm)
                    peaks_info.append(((h, k, l), two_theta_peak, I_peak))
    
        # 強度を規格化
        intensity = (intensity / intensity.max()) * 1000
    
        return two_theta, intensity, peaks_info
    
    # シリコンのXRDパターンをシミュレーション
    a_Si = 5.4310
    diamond_positions = [
        (0, 0, 0), (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5),
        (0.25, 0.25, 0.25), (0.75, 0.75, 0.25),
        (0.75, 0.25, 0.75), (0.25, 0.75, 0.75)
    ]
    f_Si = [14] * 8  # シリコンの原子番号
    
    two_theta, intensity, peaks = simulate_xrd_pattern(
        'Silicon (Si)',
        a=a_Si,
        crystal_system='cubic',
        positions=diamond_positions,
        f_atoms=f_Si,
        fwhm=0.15
    )
    
    # グラフ表示
    fig, ax = plt.subplots(figsize=(14, 6))
    
    ax.plot(two_theta, intensity, 'b-', linewidth=1.5, label='シミュレーション')
    ax.fill_between(two_theta, 0, intensity, alpha=0.2, color='blue')
    
    # 主要ピークにラベルを付ける
    peaks_sorted = sorted(peaks, key=lambda x: x[2], reverse=True)[:6]
    for (h, k, l), pos, I in peaks_sorted:
        ax.annotate(f'({h}{k}{l})',
                   xy=(pos, I * 1000 / intensity.max()),
                   xytext=(pos, I * 1000 / intensity.max() + 80),
                   ha='center', fontsize=10,
                   arrowprops=dict(arrowstyle='->', color='red', lw=1))
    
    ax.set_xlim(20, 100)
    ax.set_ylim(0, 1100)
    ax.set_xlabel('2θ (度)', fontsize=14, fontweight='bold')
    ax.set_ylabel('強度（任意単位）', fontsize=14, fontweight='bold')
    ax.set_title('シリコン（Si）の粉末XRDパターン（シミュレーション）',
                fontsize=16, fontweight='bold')
    ax.grid(axis='both', alpha=0.3)
    ax.legend(fontsize=12)
    
    plt.tight_layout()
    plt.savefig('si_powder_xrd_simulation.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("粉末XRDシミュレーションを保存しました: si_powder_xrd_simulation.png")
    

### 4.3 実測データの読み込みと解析

#### コード例6：XRDデータの読み込みとピーク検出

実際のXRD測定データ（テキストファイル）を読み込み、ピークを自動検出します：
    
    
    from scipy.signal import find_peaks
    from scipy.optimize import curve_fit
    
    def read_xrd_data(filename):
        """
        XRDデータファイルを読み込む
    
        一般的なフォーマット：
        2theta  Intensity
        20.0    150.2
        20.1    152.3
        ...
    
        Parameters:
        -----------
        filename : str
            データファイルのパス
    
        Returns:
        --------
        two_theta, intensity : arrays
        """
        try:
            data = np.loadtxt(filename, skiprows=1)  # ヘッダー行をスキップ
            two_theta = data[:, 0]
            intensity = data[:, 1]
            return two_theta, intensity
        except FileNotFoundError:
            print(f"ファイル {filename} が見つかりません。")
            print("サンプルデータを生成します。")
            # サンプルデータを生成
            return simulate_xrd_pattern('Sample', a=5.0, fwhm=0.3)[:2]
    
    def detect_peaks_in_xrd(two_theta, intensity, prominence=50, distance=10):
        """
        XRDパターンからピークを検出
    
        Parameters:
        -----------
        two_theta : array
            2θ データ
        intensity : array
            強度データ
        prominence : float
            ピーク検出の閾値（突出度）
        distance : int
            ピーク間の最小距離（データポイント数）
    
        Returns:
        --------
        peak_positions, peak_intensities : arrays
        """
        peaks_idx, properties = find_peaks(intensity,
                                           prominence=prominence,
                                           distance=distance)
    
        peak_positions = two_theta[peaks_idx]
        peak_intensities = intensity[peaks_idx]
        peak_prominences = properties['prominences']
    
        return peak_positions, peak_intensities, peak_prominences
    
    def analyze_xrd_data():
        """実測XRDデータの解析デモンストレーション"""
    
        # データ読み込み（実際のファイルがない場合はシミュレーション）
        two_theta, intensity = read_xrd_data('sample_xrd.txt')
    
        # ピーク検出
        peak_pos, peak_int, peak_prom = detect_peaks_in_xrd(
            two_theta, intensity,
            prominence=100,
            distance=20
        )
    
        print("=== 検出されたピーク ===\n")
        print(f"{'ピーク番号':<12} {'2θ (度)':<12} {'強度':<15} {'d間隔 (Å)':<12}")
        print("-" * 60)
    
        wavelength = 1.5406  # Cu Kα
    
        for i, (pos, intensity_val) in enumerate(zip(peak_pos, peak_int), 1):
            # ブラッグの法則から d 間隔を逆算
            theta = np.radians(pos / 2)
            d_spacing = wavelength / (2 * np.sin(theta))
    
            print(f"{i:<12} {pos:10.2f}  {intensity_val:12.1f}  {d_spacing:10.4f}")
    
        # グラフ表示
        fig, ax = plt.subplots(figsize=(14, 7))
    
        # XRDパターンをプロット
        ax.plot(two_theta, intensity, 'b-', linewidth=1.5, label='測定データ')
    
        # 検出されたピークをマーク
        ax.plot(peak_pos, peak_int, 'ro', markersize=8, label='検出ピーク')
    
        # ピーク番号を表示
        for i, (pos, int_val) in enumerate(zip(peak_pos, peak_int), 1):
            ax.annotate(f'{i}',
                       xy=(pos, int_val),
                       xytext=(pos, int_val + 80),
                       ha='center', fontsize=10, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    
        ax.set_xlabel('2θ (度)', fontsize=14, fontweight='bold')
        ax.set_ylabel('強度（任意単位）', fontsize=14, fontweight='bold')
        ax.set_title('XRDパターンとピーク検出', fontsize=16, fontweight='bold')
        ax.legend(fontsize=12)
        ax.grid(axis='both', alpha=0.3)
    
        plt.tight_layout()
        plt.savefig('xrd_peak_detection.png', dpi=150, bbox_inches='tight')
        plt.show()
        print("\nピーク検出結果を保存しました: xrd_peak_detection.png")
    
    # 実行
    analyze_xrd_data()
    

### 4.4 ピークフィッティング

#### コード例7：ガウス・ローレンツ関数によるピークフィッティング

ピーク形状を数学的にフィッティングして、正確なピーク位置と幅を決定します：
    
    
    def pseudo_voigt(x, amplitude, center, fwhm, eta):
        """
        擬フォークト関数（ガウス成分とローレンツ成分の混合）
    
        Parameters:
        -----------
        x : array
            データポイント
        amplitude : float
            ピーク振幅
        center : float
            ピーク中心
        fwhm : float
            半値全幅
        eta : float
            ローレンツ成分の割合（0: ガウス、1: ローレンツ）
    
        Returns:
        --------
        array : ピーク形状
        """
        # ガウス成分
        sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
        gaussian = np.exp(-((x - center)**2) / (2 * sigma**2))
    
        # ローレンツ成分
        gamma = fwhm / 2
        lorentzian = gamma**2 / ((x - center)**2 + gamma**2)
    
        # 混合
        return amplitude * (eta * lorentzian + (1 - eta) * gaussian)
    
    def fit_single_peak(two_theta, intensity, peak_center, window=2.0):
        """
        単一ピークをフィッティング
    
        Parameters:
        -----------
        two_theta : array
            2θ データ
        intensity : array
            強度データ
        peak_center : float
            ピークの概算中心位置
        window : float
            フィッティング範囲（±window 度）
    
        Returns:
        --------
        popt : array
            最適化されたパラメータ [amplitude, center, fwhm, eta]
        pcov : array
            共分散行列
        """
        # フィッティング範囲を抽出
        mask = (two_theta >= peak_center - window) & (two_theta <= peak_center + window)
        x_data = two_theta[mask]
        y_data = intensity[mask]
    
        # 初期推定値
        amplitude_init = np.max(y_data) - np.min(y_data)
        center_init = peak_center
        fwhm_init = 0.2
        eta_init = 0.5
    
        p0 = [amplitude_init, center_init, fwhm_init, eta_init]
    
        # 境界条件
        bounds = ([0, peak_center - 1, 0.05, 0],
                  [amplitude_init * 2, peak_center + 1, 1.0, 1])
    
        try:
            popt, pcov = curve_fit(pseudo_voigt, x_data, y_data, p0=p0, bounds=bounds)
            return popt, pcov, x_data, y_data
        except RuntimeError:
            print(f"ピーク {peak_center:.2f}° のフィッティングに失敗しました。")
            return None, None, x_data, y_data
    
    def demo_peak_fitting():
        """ピークフィッティングのデモンストレーション"""
    
        # サンプルデータ生成
        two_theta, intensity = simulate_xrd_pattern('Sample', a=5.4, fwhm=0.2)[:2]
    
        # ピーク検出
        peak_pos, _, _ = detect_peaks_in_xrd(two_theta, intensity, prominence=100)
    
        # 最も強いピーク3つをフィッティング
        strongest_peaks = sorted(zip(peak_pos, intensity[np.isin(two_theta, peak_pos)]),
                                key=lambda x: x[1], reverse=True)[:3]
    
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
        for ax, (peak_center, _) in zip(axes, strongest_peaks):
            result = fit_single_peak(two_theta, intensity, peak_center, window=2.5)
    
            if result[0] is not None:
                popt, pcov, x_fit, y_fit = result
                amplitude, center, fwhm, eta = popt
    
                # フィッティング曲線
                x_fine = np.linspace(x_fit.min(), x_fit.max(), 500)
                y_fine = pseudo_voigt(x_fine, *popt)
    
                # プロット
                ax.plot(x_fit, y_fit, 'bo', markersize=4, label='測定データ')
                ax.plot(x_fine, y_fine, 'r-', linewidth=2, label='フィッティング')
    
                # 結果表示
                textstr = f'中心: {center:.3f}°\nFWHM: {fwhm:.3f}°\nη: {eta:.2f}'
                ax.text(0.05, 0.95, textstr, transform=ax.transAxes,
                       fontsize=10, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
                ax.set_xlabel('2θ (度)', fontsize=12, fontweight='bold')
                ax.set_ylabel('強度', fontsize=12, fontweight='bold')
                ax.set_title(f'ピーク @ {center:.1f}°', fontsize=13, fontweight='bold')
                ax.legend(fontsize=10)
                ax.grid(alpha=0.3)
    
        plt.tight_layout()
        plt.savefig('xrd_peak_fitting.png', dpi=150, bbox_inches='tight')
        plt.show()
        print("ピークフィッティング結果を保存しました: xrd_peak_fitting.png")
    
    # 実行
    demo_peak_fitting()
    

## 5\. pymatgenを使った高度なXRD解析

### 5.1 pymatgenによるXRDパターン生成

**pymatgen** は材料科学のための強力なPythonライブラリで、 結晶構造からXRDパターンを自動生成できます。 

#### コード例8：pymatgenでのXRDパターン生成と比較
    
    
    try:
        from pymatgen.core import Structure, Lattice
        from pymatgen.analysis.diffraction.xrd import XRDCalculator
        PYMATGEN_AVAILABLE = True
    except ImportError:
        print("pymatgenがインストールされていません。")
        print("インストール: pip install pymatgen")
        PYMATGEN_AVAILABLE = False
    
    def generate_xrd_with_pymatgen():
        """pymatgenを使ってXRDパターンを生成"""
    
        if not PYMATGEN_AVAILABLE:
            print("pymatgenがインストールされていないため、この例はスキップされます。")
            return
    
        # シリコンの構造を定義
        lattice = Lattice.cubic(5.4310)
        species = ['Si', 'Si', 'Si', 'Si', 'Si', 'Si', 'Si', 'Si']
        coords = [
            [0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5],
            [0.25, 0.25, 0.25], [0.75, 0.75, 0.25],
            [0.75, 0.25, 0.75], [0.25, 0.75, 0.75]
        ]
    
        si_structure = Structure(lattice, species, coords)
    
        print("=== pymatgenによるXRDパターン生成 ===\n")
        print(f"結晶構造: {si_structure.composition}")
        print(f"空間群: {si_structure.get_space_group_info()}\n")
    
        # XRD計算機の初期化
        calculator = XRDCalculator(wavelength='CuKa')  # Cu Kα線
    
        # XRDパターンを計算
        pattern = calculator.get_pattern(si_structure, two_theta_range=(20, 100))
    
        print(f"{'2θ (度)':<12} {'d間隔 (Å)':<15} {'(hkl)':<15} {'相対強度':<12}")
        print("-" * 60)
    
        for i in range(len(pattern.x)):
            two_theta = pattern.x[i]
            intensity = pattern.y[i]
            hkl = pattern.hkls[i][0]['hkl']  # 最初のhklを取得
            d_spacing = pattern.d_hkls[i]
    
            print(f"{two_theta:10.2f}  {d_spacing:12.4f}  {str(hkl):<15} {intensity:10.1f}")
    
        # グラフ表示
        fig, ax = plt.subplots(figsize=(14, 7))
    
        # 棒グラフとしてプロット
        ax.vlines(pattern.x, 0, pattern.y, colors='blue', linewidth=2, label='pymatgen')
    
        # ピークにhklラベルを付ける
        for i, (two_theta, intensity, hkls_data) in enumerate(zip(pattern.x, pattern.y, pattern.hkls)):
            if intensity > 20:  # 強度が20以上のピークのみラベル表示
                hkl = hkls_data[0]['hkl']
                ax.text(two_theta, intensity + 5, f'({hkl[0]}{hkl[1]}{hkl[2]})',
                       rotation=90, va='bottom', ha='center', fontsize=9)
    
        ax.set_xlim(20, 100)
        ax.set_ylim(0, max(pattern.y) * 1.15)
        ax.set_xlabel('2θ (度)', fontsize=14, fontweight='bold')
        ax.set_ylabel('相対強度', fontsize=14, fontweight='bold')
        ax.set_title('シリコン（Si）のXRDパターン - pymatgen生成',
                    fontsize=16, fontweight='bold')
        ax.legend(fontsize=12)
        ax.grid(axis='both', alpha=0.3)
    
        plt.tight_layout()
        plt.savefig('si_xrd_pymatgen.png', dpi=150, bbox_inches='tight')
        plt.show()
        print("\npymatgen XRDパターンを保存しました: si_xrd_pymatgen.png")
    
        # 複数の材料を比較
        compare_materials_xrd()
    
    def compare_materials_xrd():
        """複数の材料のXRDパターンを比較"""
    
        if not PYMATGEN_AVAILABLE:
            return
    
        # 材料の定義
        materials = {
            'Si (Diamond)': Structure(
                Lattice.cubic(5.4310),
                ['Si']*8,
                [[0,0,0], [0.5,0.5,0], [0.5,0,0.5], [0,0.5,0.5],
                 [0.25,0.25,0.25], [0.75,0.75,0.25], [0.75,0.25,0.75], [0.25,0.75,0.75]]
            ),
            'Al (FCC)': Structure(
                Lattice.cubic(4.0495),
                ['Al']*4,
                [[0,0,0], [0.5,0.5,0], [0.5,0,0.5], [0,0.5,0.5]]
            ),
            'Fe (BCC)': Structure(
                Lattice.cubic(2.8665),
                ['Fe']*2,
                [[0,0,0], [0.5,0.5,0.5]]
            )
        }
    
        calculator = XRDCalculator(wavelength='CuKa')
    
        fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    
        for ax, (name, structure) in zip(axes, materials.items()):
            pattern = calculator.get_pattern(structure, two_theta_range=(20, 100))
    
            # 棒グラフ
            ax.vlines(pattern.x, 0, pattern.y, colors='darkblue', linewidth=2.5)
    
            # ピークラベル
            for two_theta, intensity, hkls_data in zip(pattern.x, pattern.y, pattern.hkls):
                if intensity > 15:
                    hkl = hkls_data[0]['hkl']
                    ax.text(two_theta, intensity + 3, f'({hkl[0]}{hkl[1]}{hkl[2]})',
                           rotation=90, va='bottom', ha='center', fontsize=9)
    
            ax.set_xlim(20, 100)
            ax.set_ylim(0, 110)
            ax.set_ylabel('相対強度', fontsize=12, fontweight='bold')
            ax.set_title(name, fontsize=14, fontweight='bold', loc='left')
            ax.grid(axis='x', alpha=0.3)
    
        axes[-1].set_xlabel('2θ (度)', fontsize=14, fontweight='bold')
    
        plt.tight_layout()
        plt.savefig('materials_xrd_comparison.png', dpi=150, bbox_inches='tight')
        plt.show()
        print("材料比較グラフを保存しました: materials_xrd_comparison.png")
    
    # 実行
    generate_xrd_with_pymatgen()
    

#### pymatgenの利点

  * **構造データベース連携** ：Materials Projectなどから直接構造を取得可能
  * **自動的な構造因子計算** ：原子散乱因子や温度因子を含む正確な計算
  * **空間群の自動認識** ：対称性を考慮した正確なパターン生成
  * **相同定機能** ：実測パターンとの比較による未知試料の同定

## 6\. リートベルト解析入門

### 6.1 リートベルト法とは

**リートベルト解析（Rietveld refinement）** は、 粉末XRDパターン全体を結晶構造モデルでフィッティングする手法です。 1969年にフーゴー・リートベルトによって開発されました。 

#### リートベルト法で得られる情報

  * **格子定数** ：高精度（±0.0001 Å以下）
  * **原子座標** ：単位格子内の原子位置
  * **占有率** ：原子サイトの占有率（合金や欠陥の評価）
  * **温度因子** ：原子の熱振動の大きさ
  * **結晶子サイズ** ：ピーク幅からの推定
  * **微小ひずみ** ：格子の歪み
  * **相の定量** ：多相混合物の各相の重量分率

### 6.2 リートベルト法の原理

リートベルト法では、以下の関数を最小二乗法で最適化します：

$$ S = \sum_i w_i (y_{i,\text{obs}} - y_{i,\text{calc}})^2 $$ 

ここで：

  * **y i,obs**：i番目の測定点での観測強度
  * **y i,calc**：i番目の測定点での計算強度
  * **w i**：重み（通常は 1/yi,obs）

計算強度は以下のように表されます：

$$ y_{i,\text{calc}} = \text{scale} \sum_{K} L_K |F_K|^2 \Phi(2\theta_i - 2\theta_K) P_K A + y_{i,\text{bg}} $$ 

#### リートベルト解析の注意点

リートベルト法は**構造モデルありき** の手法です。 初期構造モデルが大きく間違っていると、正しい解に収束しません。 通常は、既知の類似構造や単結晶XRDデータから初期モデルを作成します。 

### 6.3 リートベルト解析のワークフロー
    
    
    ```mermaid
    flowchart TD
                    A[粉末XRDデータ測定] --> B[ピーク同定と相の特定]
                    B --> C[初期構造モデル作成]
                    C --> D[バックグラウンド設定]
                    D --> E[格子定数の精密化]
                    E --> F[プロファイル形状の精密化]
                    F --> G[構造パラメータの精密化]
                    G --> H{適合度チェック}
                    H -->|良好| I[結果の検証と報告]
                    H -->|不良| J[モデル修正]
                    J --> D
    
                    style A fill:#e3f2fd
                    style B fill:#e3f2fd
                    style C fill:#fff3e0
                    style I fill:#e8f5e9
                    style J fill:#ffebee
    ```

## 7\. 演習問題

### 演習1：ブラッグの法則の応用

銅（Cu）の面心立方構造（FCC）で格子定数 a = 3.615 Å とします。 Cu Kα線（λ = 1.5406 Å）を用いた粉末XRD測定で、 以下の(hkl)面からの回折ピークは何度（2θ）に観測されますか？ 

  1. (111)面
  2. (200)面
  3. (220)面

また、FCC構造の消滅則により、(100)ピークは観測されるか説明しなさい。

解答を見る

**面間隔の計算：**

  1. d111 = 3.615 / √3 = 2.087 Å
  2. d200 = 3.615 / √4 = 1.808 Å
  3. d220 = 3.615 / √8 = 1.278 Å

**ブラッグ角の計算（λ = 2d sinθ）：**

  1. 2θ111 = 2 × arcsin(1.5406/(2×2.087)) ≈ **43.3°**
  2. 2θ200 = 2 × arcsin(1.5406/(2×1.808)) ≈ **50.4°**
  3. 2θ220 = 2 × arcsin(1.5406/(2×1.278)) ≈ **74.1°**

**(100)面について：**

FCC構造の消滅則は「h, k, l がすべて偶数またはすべて奇数のみ観測」です。 (100)は h=1（奇数）、k=0（偶数）、l=0（偶数）なので混合となり、 構造因子 F100 = 0 となります。したがって**観測されません** 。 

### 演習2：構造因子と消滅則

あるXRD測定で以下のピークが観測されました： (110), (200), (211), (220), (310), (222), (321), (400) 

この材料は単純立方（SC）、体心立方（BCC）、面心立方（FCC）のどれですか？ 消滅則から判定しなさい。 

解答を見る

各指数で h+k+l の和を確認：

  * (110): 1+1+0 = 2 (偶数)
  * (200): 2+0+0 = 2 (偶数)
  * (211): 2+1+1 = 4 (偶数)
  * (220): 2+2+0 = 4 (偶数)
  * (310): 3+1+0 = 4 (偶数)
  * (222): 2+2+2 = 6 (偶数)
  * (321): 3+2+1 = 6 (偶数)
  * (400): 4+0+0 = 4 (偶数)

すべてのピークで h+k+l が偶数です。これは**BCC（体心立方）** の消滅則と一致します。 

もしFCCなら、(210)や(221)などの混合指数が消滅するはずですが、 観測されたピークにそのような規則性はありません。 SCなら(100)などすべてのピークが観測されるはずです。 

**答え：BCC（体心立方）**

### 演習3：d間隔からの結晶同定

未知試料のXRDパターンから、以下の d 間隔（Å）が得られました： 3.35, 2.46, 2.13, 1.91, 1.80 

この試料は以下のどの材料と考えられますか？格子定数から判定しなさい。 

  * A) NaCl（岩塩構造、FCC、a = 5.64 Å）
  * B) Si（ダイヤモンド構造、a = 5.43 Å）
  * C) グラファイト（六方晶、a = 2.46 Å、c = 6.71 Å）

解答を見る

各材料の代表的な d 間隔を計算：

**A) NaCl (FCC, a=5.64 Å):**

  * d111 = 5.64/√3 = 3.26 Å
  * d200 = 5.64/√4 = 2.82 Å
  * d220 = 5.64/√8 = 2.00 Å

**B) Si (a=5.43 Å):**

  * d111 = 5.43/√3 = 3.14 Å
  * d220 = 5.43/√8 = 1.92 Å

**C) グラファイト (六方晶, a=2.46, c=6.71 Å):**

  * d002 = c/2 = 3.35 Å ✓
  * d100 = a√(3/4) = 2.13 Å ✓
  * d004 = c/4 = 1.68 Å

**答え：C) グラファイト**

最大 d 間隔の 3.35 Å がグラファイトの特徴的な (002) 面間隔と一致します。 これはグラファイト層間距離に対応する重要なピークです。 

### 演習4：プログラミング課題

以下の条件で架空の材料のXRDパターンをシミュレートし、 主要なピーク（上位5つ）の位置と相対強度を表示するプログラムを作成しなさい： 

  * 結晶系：立方晶（FCC構造）
  * 格子定数：a = 4.00 Å
  * X線波長：Cu Kα線（λ = 1.5406 Å）
  * 測定範囲：2θ = 20° 〜 90°

ヒント

コード例1と3を組み合わせて使用します。FCC構造の原子位置と構造因子計算、 ブラッグの法則による回折角計算、多重度を考慮した強度計算を実装します。 

完成したプログラムは、(111), (200), (220), (311), (222) などのピークを 正しい角度と相対強度で表示するはずです。 

## まとめ

この章では、X線回折の原理と実際の解析手法を学びました：

#### 重要ポイント

  * **ブラッグの法則 nλ = 2d sinθ** は回折が起こる必要条件
  * **構造因子 F hkl**がゼロの場合、ブラッグの法則を満たしても回折しない（消滅則）
  * **消滅則** から結晶構造（SC, BCC, FCC等）を推定できる
  * **回折強度** は構造因子、多重度、ローレンツ偏光因子、温度因子など多くの因子に依存
  * **粉末XRD** は材料同定、相分析、格子定数決定に広く使われる
  * **ピークフィッティング** により正確なピーク位置と幅を決定できる
  * **pymatgen** を使うと、結晶構造から自動的にXRDパターンを生成できる
  * **リートベルト解析** はパターン全体をフィッティングし、構造パラメータを精密化する手法

次章では、**結晶構造の可視化と解析** を学び、 実際の材料データベースから構造を取得して解析する実践的なスキルを習得します。 

### 免責事項

  * 本コンテンツは教育・研究・情報提供のみを目的としており、専門的な助言(法律・会計・技術的保証など)を提供するものではありません。
  * 本コンテンツおよび付随するCode examplesは「現状有姿(AS IS)」で提供され、明示または黙示を問わず、商品性、特定目的適合性、権利非侵害、正確性・完全性、動作・安全性等いかなる保証もしません。
  * 外部リンク、第三者が提供するデータ・ツール・ライブラリ等の内容・可用性・安全性について、作成者および東北大学は一切の責任を負いません。
  * 本コンテンツの利用・実行・解釈により直接的・間接的・付随的・特別・結果的・懲罰的損害が生じた場合でも、適用法で許容される最大限の範囲で、作成者および東北大学は責任を負いません。
  * 本コンテンツの内容は、予告なく変更・更新・提供停止されることがあります。
  * 本コンテンツの著作権・ライセンスは明記された条件(例: CC BY 4.0)に従います。当該ライセンスは通常、無保証条項を含みます。
