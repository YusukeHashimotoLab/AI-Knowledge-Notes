---
title: "第5章: 三元系相図とCALPHAD法"
chapter_title: "第5章: 三元系相図とCALPHAD法"
subtitle: Fe-Cr-Ni、Al-Cu-Mg系の三元系相図とCALPHAD法による相図計算の原理
---

🌐 JP | [🇬🇧 EN](<../../../en/MS/materials-thermodynamics-introduction/chapter-5.html>) | Last sync: 2025-11-16

## 学習目標

この章では、実用材料で不可欠な**三元系相図** の読み方と、計算材料科学の基盤である**CALPHAD法（CALculation of PHAse Diagrams）** の原理を学びます。三元系は二元系よりも複雑ですが、ステンレス鋼（Fe-Cr-Ni）、高強度アルミニウム合金（Al-Cu-Mg）など、工業材料の多くは三元系以上の多元系です。

#### この章で習得するスキル

  * ギブスの三角形（Gibbs triangle）による組成表現
  * アイソサーマルセクション（等温断面図）の読み方
  * 液相面投影図（liquidus projection）の解析
  * 垂直断面図（pseudo-binary section）の作成と解釈
  * 三元系共晶点・包晶点の決定
  * CALPHAD法の原理と熱力学データベースの構造
  * Redlich-Kister式による多元系モデリング
  * 相図計算ワークフローの実践

#### 💡 三元系相図とCALPHAD法の重要性

三元系相図は、3成分合金の平衡状態を3次元空間（組成2軸 + 温度軸）で表現します。しかし3D図は読みにくいため、通常は等温断面図、液相面投影図、垂直断面図などの2D断面で可視化します。CALPHAD法は、熱力学データベースを用いて相図を計算する手法で、実験が困難な領域の相図予測や、新合金の設計に不可欠です。

## 1\. ギブスの三角形による組成表現

三元系A-B-Cの組成は、**ギブスの三角形** （Gibbs triangle）で表現します。正三角形の各頂点が純成分（A, B, C）に対応し、各辺が二元系（A-B, B-C, C-A）を表します。

### 1.1 三角座標系の原理

  * **頂点** : A（100% A）、B（100% B）、C（100% C）
  * **辺** : AB辺（C = 0%）、BC辺（A = 0%）、CA辺（B = 0%）
  * **内部の点** : 三元合金（A + B + C = 100%）
  * **等組成線** : 各成分の等量線が辺に平行に引かれる

#### 組成の読み取り方

点Pの組成（\\(x_A, x_B, x_C\\)）は、以下の手順で読み取ります：

  1. 点PからBC辺への垂線の長さが\\(x_A\\)に比例
  2. 点PからCA辺への垂線の長さが\\(x_B\\)に比例
  3. 点PからAB辺への垂線の長さが\\(x_C\\)に比例
  4. \\(x_A + x_B + x_C = 1\\)（または100%）

#### コード例1: ギブスの三角形での組成表現
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    
    def ternary_to_cartesian(a, b, c):
        """三角座標をデカルト座標に変換"""
        x = 0.5 * (2*b + c) / (a + b + c)
        y = (np.sqrt(3)/2) * c / (a + b + c)
        return x, y
    
    # ギブスの三角形の描画
    fig, ax = plt.subplots(figsize=(10, 9))
    
    # 三角形の頂点（A, B, C）
    vertices = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3)/2]])
    triangle = Polygon(vertices, fill=False, edgecolor='black', linewidth=2)
    ax.add_patch(triangle)
    
    # 頂点ラベル
    ax.text(-0.05, -0.05, 'A (Fe)', fontsize=14, fontweight='bold')
    ax.text(1.05, -0.05, 'B (Cr)', fontsize=14, fontweight='bold')
    ax.text(0.5, np.sqrt(3)/2 + 0.05, 'C (Ni)', fontsize=14, fontweight='bold', ha='center')
    
    # 等組成線（グリッド）を描画
    for i in range(1, 10):
        t = i / 10
        # A成分の等量線（BC辺に平行）
        x1, y1 = ternary_to_cartesian(t, 1-t, 0)
        x2, y2 = ternary_to_cartesian(t, 0, 1-t)
        ax.plot([x1, x2], [y1, y2], 'gray', linewidth=0.5, alpha=0.5)
    
        # B成分の等量線（CA辺に平行）
        x1, y1 = ternary_to_cartesian(1-t, t, 0)
        x2, y2 = ternary_to_cartesian(0, t, 1-t)
        ax.plot([x1, x2], [y1, y2], 'gray', linewidth=0.5, alpha=0.5)
    
        # C成分の等量線（AB辺に平行）
        x1, y1 = ternary_to_cartesian(1-t, 0, t)
        x2, y2 = ternary_to_cartesian(0, 1-t, t)
        ax.plot([x1, x2], [y1, y2], 'gray', linewidth=0.5, alpha=0.5)
    
    # サンプル組成点をプロット（Fe-18Cr-8Ni: SUS304ステンレス鋼）
    a_sample, b_sample, c_sample = 0.74, 0.18, 0.08  # モル分率
    x_sample, y_sample = ternary_to_cartesian(a_sample, b_sample, c_sample)
    ax.plot(x_sample, y_sample, 'ro', markersize=10, label='SUS304 (Fe-18Cr-8Ni)')
    ax.text(x_sample + 0.03, y_sample, 'SUS304', fontsize=11, color='red')
    
    # 他の重要な組成点
    compositions = {
        'SUS316': (0.68, 0.17, 0.12),  # Fe-17Cr-12Ni
        'SUS430': (0.83, 0.17, 0.00),  # Fe-17Cr (フェライト系)
    }
    
    for name, (a, b, c) in compositions.items():
        x, y = ternary_to_cartesian(a, b, c)
        ax.plot(x, y, 'bs', markersize=8)
        ax.text(x + 0.03, y, name, fontsize=10, color='blue')
    
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 1.0)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.legend(loc='upper right')
    ax.set_title('ギブスの三角形: Fe-Cr-Ni三元系の組成表現', fontsize=15, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('gibbs_triangle.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("📌 組成の確認:")
    print(f"SUS304: Fe={a_sample*100:.1f}%, Cr={b_sample*100:.1f}%, Ni={c_sample*100:.1f}%")
    print(f"合計: {(a_sample + b_sample + c_sample)*100:.1f}%")

#### 💡 実用例: ステンレス鋼の組成表現

ステンレス鋼は、Fe-Cr-Ni三元系の代表例です。SUS304（Fe-18Cr-8Ni）はオーステナイト系、SUS430（Fe-17Cr）はフェライト系ステンレスです。ギブスの三角形上で、これらの組成が相図のどの相領域にあるかを確認することで、室温での結晶構造（オーステナイトFCCまたはフェライトBCC）を予測できます。

## 2\. アイソサーマルセクション（等温断面図）

**アイソサーマルセクション** は、特定温度での三元系の相平衡を示す断面図です。ギブスの三角形上に、各相領域と相境界線が描かれます。

### 2.1 等温断面図の読み方

  * **単相領域** : α、β、γ、L（液相）など、1つの相のみが安定
  * **二相領域** : α+β、L+αなど、2つの相が共存
  * **三相領域** : α+β+γなど、3つの相が共存（三角形の領域として描かれる）
  * **タイライン（tie-line）** : 二相領域内で平衡する2相の組成を結ぶ直線
  * **タイトライアングル（tie-triangle）** : 三相領域内で平衡する3相の組成を結ぶ三角形

#### コード例2: アイソサーマルセクション（等温断面図）の作成
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    from matplotlib.collections import LineCollection
    
    def ternary_to_cartesian(a, b, c):
        """三角座標をデカルト座標に変換"""
        x = 0.5 * (2*b + c) / (a + b + c)
        y = (np.sqrt(3)/2) * c / (a + b + c)
        return x, y
    
    # 1200℃でのFe-Cr-Ni系の簡易的な等温断面図（模式図）
    fig, ax = plt.subplots(figsize=(11, 10))
    
    # 三角形の頂点
    vertices = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3)/2]])
    triangle = Polygon(vertices, fill=False, edgecolor='black', linewidth=2)
    ax.add_patch(triangle)
    
    # 頂点ラベル
    ax.text(-0.05, -0.05, 'Fe', fontsize=14, fontweight='bold')
    ax.text(1.05, -0.05, 'Cr', fontsize=14, fontweight='bold')
    ax.text(0.5, np.sqrt(3)/2 + 0.05, 'Ni', fontsize=14, fontweight='bold', ha='center')
    
    # 相領域の定義（簡略化）
    # 液相領域（L）
    liquid_region = np.array([
        ternary_to_cartesian(0.2, 0.3, 0.5),
        ternary_to_cartesian(0.1, 0.5, 0.4),
        ternary_to_cartesian(0.15, 0.6, 0.25),
        ternary_to_cartesian(0.25, 0.45, 0.3),
    ])
    liquid_patch = Polygon(liquid_region, alpha=0.3, facecolor='lightblue',
                           edgecolor='blue', linewidth=1.5, label='液相 (L)')
    ax.add_patch(liquid_patch)
    
    # オーステナイト相領域（γ-FCC）
    austenite_region = np.array([
        ternary_to_cartesian(0.7, 0.1, 0.2),
        ternary_to_cartesian(0.5, 0.15, 0.35),
        ternary_to_cartesian(0.6, 0.05, 0.35),
        ternary_to_cartesian(0.8, 0.05, 0.15),
    ])
    austenite_patch = Polygon(austenite_region, alpha=0.3, facecolor='lightgreen',
                              edgecolor='green', linewidth=1.5, label='γ (FCC)')
    ax.add_patch(austenite_patch)
    
    # フェライト相領域（α-BCC）
    ferrite_region = np.array([
        ternary_to_cartesian(0.9, 0.1, 0.0),
        ternary_to_cartesian(0.8, 0.2, 0.0),
        ternary_to_cartesian(0.7, 0.25, 0.05),
        ternary_to_cartesian(0.85, 0.12, 0.03),
    ])
    ferrite_patch = Polygon(ferrite_region, alpha=0.3, facecolor='lightyellow',
                            edgecolor='orange', linewidth=1.5, label='α (BCC)')
    ax.add_patch(ferrite_patch)
    
    # L + γ 二相領域
    L_gamma_region = np.array([
        ternary_to_cartesian(0.25, 0.45, 0.3),
        ternary_to_cartesian(0.35, 0.3, 0.35),
        ternary_to_cartesian(0.5, 0.15, 0.35),
        ternary_to_cartesian(0.2, 0.3, 0.5),
    ])
    L_gamma_patch = Polygon(L_gamma_region, alpha=0.2, facecolor='cyan',
                            edgecolor='blue', linestyle='--', linewidth=1, label='L + γ')
    ax.add_patch(L_gamma_patch)
    
    # タイライン（tie-line）の例
    tie_lines = [
        [ternary_to_cartesian(0.3, 0.35, 0.35), ternary_to_cartesian(0.45, 0.2, 0.35)],
        [ternary_to_cartesian(0.25, 0.4, 0.35), ternary_to_cartesian(0.48, 0.18, 0.34)],
    ]
    
    for tie_line in tie_lines:
        xs, ys = zip(*tie_line)
        ax.plot(xs, ys, 'k--', linewidth=1, alpha=0.6)
    
    # 相ラベル
    ax.text(*ternary_to_cartesian(0.15, 0.45, 0.4), 'L', fontsize=12, fontweight='bold', ha='center')
    ax.text(*ternary_to_cartesian(0.65, 0.1, 0.25), 'γ', fontsize=12, fontweight='bold', ha='center')
    ax.text(*ternary_to_cartesian(0.82, 0.15, 0.03), 'α', fontsize=12, fontweight='bold', ha='center')
    ax.text(*ternary_to_cartesian(0.35, 0.3, 0.35), 'L+γ', fontsize=10, style='italic', ha='center')
    
    # 代表的な組成点
    compositions = {
        'SUS304': (0.74, 0.18, 0.08),
        'SUS316': (0.68, 0.17, 0.15),
    }
    
    for name, (a, b, c) in compositions.items():
        x, y = ternary_to_cartesian(a, b, c)
        ax.plot(x, y, 'ro', markersize=8)
        ax.text(x + 0.03, y, name, fontsize=10, color='red')
    
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 1.0)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.legend(loc='upper right', fontsize=10)
    ax.set_title('Fe-Cr-Ni三元系の等温断面図（1200℃, 模式図）', fontsize=15, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('isothermal_section.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("📌 1200℃での相状態:")
    print("・ SUS304 (Fe-18Cr-8Ni): γ相（オーステナイト）領域")
    print("・ 高Cr領域: α相（フェライト）が安定")
    print("・ L+γ二相領域: 液相とオーステナイトが共存")

#### 💡 タイラインとレバールール

二相領域内の組成点Pでは、タイライン上の2つの相（αとβ）が共存します。各相の分率は、二元系と同様にレバールールで計算できます：

\\[ f_\alpha = \frac{|\text{P-β}|}{|\text{α-β}|}, \quad f_\beta = \frac{|\text{P-α}|}{|\text{α-β}|} \\]

ただし、距離はギブスの三角形上での組成空間での距離です。

## 3\. 液相面投影図（Liquidus Projection）

**液相面投影図** は、液相が最初に凝固を始める温度（液相線温度）をギブスの三角形上に等高線で示した図です。冷却経路と凝固過程を理解するのに有用です。

### 3.1 液相面投影図の構成要素

  * **液相線等高線** : 同じ液相線温度を結ぶ曲線
  * **初晶線（primary crystallization lines）** : 液相から最初に晶出する相を区別する境界線
  * **共晶谷（eutectic valley）** : 液相線温度が極小となる谷線
  * **三元共晶点（ternary eutectic point）** : 液相が3つの固相に分解する不変点

#### コード例3: 液相面投影図の可視化
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    from scipy.interpolate import griddata
    
    def ternary_to_cartesian(a, b, c):
        """三角座標をデカルト座標に変換"""
        total = a + b + c
        x = 0.5 * (2*b + c) / total
        y = (np.sqrt(3)/2) * c / total
        return x, y
    
    # 液相面温度の簡易モデル（Al-Cu-Mg系を模擬）
    fig, ax = plt.subplots(figsize=(11, 10))
    
    # 三角形の頂点
    vertices = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3)/2]])
    triangle = Polygon(vertices, fill=False, edgecolor='black', linewidth=2)
    ax.add_patch(triangle)
    
    # 頂点ラベルと融点
    ax.text(-0.08, -0.05, 'Al\n(660℃)', fontsize=13, fontweight='bold', ha='right')
    ax.text(1.08, -0.05, 'Cu\n(1085℃)', fontsize=13, fontweight='bold', ha='left')
    ax.text(0.5, np.sqrt(3)/2 + 0.05, 'Mg\n(650℃)', fontsize=13, fontweight='bold', ha='center')
    
    # グリッド点での液相線温度を計算（簡易モデル）
    n_points = 100
    grid_a = []
    grid_b = []
    grid_c = []
    liquidus_temps = []
    
    for i in range(n_points):
        for j in range(n_points - i):
            k = n_points - i - j
            a, b, c = i/n_points, j/n_points, k/n_points
    
            if a + b + c > 0.99 and a + b + c < 1.01:  # 三角形内部のみ
                # 簡易的な液相線温度モデル（実際はCALPHADで計算）
                T_liquidus = (660*a + 1085*b + 650*c) - 200*a*b - 150*b*c - 100*c*a
    
                grid_a.append(a)
                grid_b.append(b)
                grid_c.append(c)
                liquidus_temps.append(T_liquidus)
    
    # デカルト座標に変換
    grid_x = []
    grid_y = []
    for a, b, c in zip(grid_a, grid_b, grid_c):
        x, y = ternary_to_cartesian(a, b, c)
        grid_x.append(x)
        grid_y.append(y)
    
    # 等高線プロット用のグリッド作成
    xi = np.linspace(0, 1, 200)
    yi = np.linspace(0, np.sqrt(3)/2, 200)
    xi_grid, yi_grid = np.meshgrid(xi, yi)
    
    # griddata補間
    zi = griddata((grid_x, grid_y), liquidus_temps, (xi_grid, yi_grid), method='cubic')
    
    # 等高線をプロット
    levels = np.arange(550, 1100, 50)
    contour = ax.contour(xi_grid, yi_grid, zi, levels=levels, colors='black', linewidths=0.5, alpha=0.6)
    ax.clabel(contour, inline=True, fontsize=8, fmt='%d℃')
    
    # 等高線を塗りつぶし
    contourf = ax.contourf(xi_grid, yi_grid, zi, levels=levels, cmap='coolwarm', alpha=0.5)
    cbar = plt.colorbar(contourf, ax=ax, label='液相線温度 (℃)', pad=0.02)
    
    # 初晶線（模式的）
    primary_lines = [
        # Al初晶領域とCu初晶領域の境界
        [ternary_to_cartesian(0.8, 0.2, 0), ternary_to_cartesian(0.5, 0.3, 0.2)],
        # Cu初晶領域とMg初晶領域の境界
        [ternary_to_cartesian(0.2, 0.8, 0), ternary_to_cartesian(0.3, 0.4, 0.3)],
        # Al初晶領域とMg初晶領域の境界
        [ternary_to_cartesian(0.7, 0, 0.3), ternary_to_cartesian(0.4, 0.1, 0.5)],
    ]
    
    for line in primary_lines:
        xs, ys = zip(*line)
        ax.plot(xs, ys, 'b-', linewidth=2, alpha=0.7)
    
    # 三元共晶点（模式的）
    eutectic_point = ternary_to_cartesian(0.5, 0.3, 0.2)
    ax.plot(*eutectic_point, 'r*', markersize=15, label='三元共晶点 (~520℃)')
    
    # 実用合金の組成
    alloys = {
        '2024': (0.935, 0.043, 0.015),  # Al-4.3Cu-1.5Mg
        '7075': (0.90, 0.016, 0.025),   # Al-1.6Cu-2.5Mg-Zn
    }
    
    for name, (a, b, c) in alloys.items():
        x, y = ternary_to_cartesian(a, b, c)
        ax.plot(x, y, 'ko', markersize=8)
        ax.text(x + 0.03, y, name, fontsize=10, color='black', fontweight='bold')
    
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 1.0)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.legend(loc='upper right', fontsize=10)
    ax.set_title('Al-Cu-Mg三元系の液相面投影図（模式図）', fontsize=15, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('liquidus_projection.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("📌 液相面投影図の読み方:")
    print("・ 等高線: 液相線温度を示す（冷却時に凝固が始まる温度）")
    print("・ 初晶線: 最初に晶出する相を区別する境界")
    print("・ 三元共晶点: L → α + β + γ の反応が起こる不変点")

#### 冷却経路の追跡

液相面投影図上で、合金の冷却経路は以下のように追跡できます：

  1. 組成点Pから冷却を開始
  2. 液相線温度に達すると、初晶（α、β、またはγ）が晶出開始
  3. 冷却が進むと、液相の組成は初晶線に沿って変化
  4. 初晶線を下り、共晶谷に到達
  5. 共晶谷を下り、三元共晶点で完全に凝固

## 4\. 垂直断面図（Pseudo-Binary Section）

**垂直断面図** は、ギブスの三角形上の特定の直線（例: A-B辺から頂点Cへの直線）に沿った組成での温度-組成図です。二元系相図と同様の見た目になります。

### 4.1 垂直断面図の用途

  * 実用合金の組成範囲での相変態を詳細に解析
  * 特定の組成比（例: A:B = 1:1）を固定した場合の第三成分Cの影響を調査
  * 二元系相図に似た形式で、三元系の局所的な振る舞いを理解

#### コード例4: 垂直断面図（Pseudo-Binary Section）
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Fe-Cr-Ni系で、Cr/Ni = 2:1の比率を固定した垂直断面図（模式図）
    # 横軸: Fe含有量 (100% → 0%)、縦軸: 温度
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Fe含有量（wt%）
    fe_content = np.linspace(100, 0, 100)
    # Cr/Ni = 2:1なので、Cr = 2(100-Fe)/3, Ni = (100-Fe)/3
    
    # 液相線と固相線（簡易モデル）
    liquidus = 1536 - 5*fe_content + 0.02*fe_content**2  # Fe側は1536℃
    solidus = 1450 - 3*fe_content + 0.015*fe_content**2
    
    # γ/α相境界（フェライトとオーステナイトの境界）
    gamma_alpha_boundary = 1400 - 6*fe_content + 0.03*fe_content**2
    
    # プロット
    ax.plot(fe_content, liquidus, 'b-', linewidth=2, label='液相線 (Liquidus)')
    ax.plot(fe_content, solidus, 'r-', linewidth=2, label='固相線 (Solidus)')
    ax.plot(fe_content, gamma_alpha_boundary, 'g--', linewidth=2, label='γ/α相境界')
    
    # 相領域のラベル
    ax.text(50, 1600, 'L (液相)', fontsize=12, ha='center', fontweight='bold')
    ax.text(50, 1500, 'L + γ', fontsize=11, ha='center', style='italic')
    ax.text(70, 1350, 'γ (FCC)', fontsize=12, ha='center', fontweight='bold', color='green')
    ax.text(30, 1250, 'α (BCC)', fontsize=12, ha='center', fontweight='bold', color='orange')
    ax.text(50, 1300, 'γ + α', fontsize=10, ha='center', style='italic')
    
    # 相領域の塗りつぶし
    ax.fill_between(fe_content, liquidus, 1700, alpha=0.2, color='lightblue', label='L')
    ax.fill_between(fe_content, solidus, liquidus, alpha=0.2, color='cyan', label='L+γ')
    ax.fill_between(fe_content, gamma_alpha_boundary, solidus,
                    where=(gamma_alpha_boundary < solidus), alpha=0.2, color='lightgreen', label='γ')
    ax.fill_between(fe_content, 1100, gamma_alpha_boundary, alpha=0.2, color='lightyellow', label='α')
    
    # SUS304の組成点（Fe-18Cr-8Ni → Cr/Ni ≈ 2.25:1）
    fe_304 = 74  # wt% Fe
    ax.axvline(fe_304, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.text(fe_304 + 2, 1150, 'SUS304', fontsize=11, color='red', fontweight='bold', rotation=90)
    
    ax.set_xlabel('Fe含有量 (wt%)', fontsize=13)
    ax.set_ylabel('温度 (℃)', fontsize=13)
    ax.set_title('Fe-Cr-Ni三元系の垂直断面図（Cr/Ni = 2:1固定, 模式図）', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 100)
    ax.set_ylim(1100, 1700)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('vertical_section.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("📌 垂直断面図の解釈:")
    print("・ SUS304 (Fe-18Cr-8Ni): 高温でγ相（オーステナイト）が安定")
    print("・ Fe含有量が増加すると、α相（フェライト）が安定化")
    print("・ L+γ二相領域: 凝固過程でγ相が晶出")

#### 💡 垂直断面図の活用例

ステンレス鋼の溶接では、凝固時の相状態が重要です。垂直断面図を用いることで、特定のCr/Ni比での凝固経路を予測し、溶接割れの原因となる有害相（σ相など）の形成を回避する組成設計が可能になります。

## 5\. 三元系共晶点と不変反応

三元系では、**三元共晶反応** \\( L \rightarrow \alpha + \beta + \gamma \\) のような不変反応が存在します。この反応は、ギブスの相律により、特定の温度と組成で起こります。

### 5.1 三元系の不変反応

反応タイプ | 反応式 | 特徴  
---|---|---  
三元共晶 | \\( L \rightarrow \alpha + \beta + \gamma \\) | 液相が3つの固相に分解  
三元包晶 | \\( L + \alpha + \beta \rightarrow \gamma \\) | 液相と2固相が反応して新しい固相を生成  
三元偏晶 | \\( L_1 \rightarrow L_2 + \alpha + \beta \\) | 液相が2つの液相と固相に分離  
準包晶 | \\( L + \alpha \rightarrow \beta + \gamma \\) | 液相と固相が反応して2つの固相を生成  
  
#### コード例5: 三元系共晶点の決定
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    from mpl_toolkits.mplot3d import Axes3D
    
    def ternary_to_cartesian(a, b, c):
        """三角座標をデカルト座標に変換"""
        total = a + b + c
        x = 0.5 * (2*b + c) / total
        y = (np.sqrt(3)/2) * c / total
        return x, y
    
    # 三元共晶反応の可視化
    fig = plt.figure(figsize=(14, 6))
    
    # 左図: ギブスの三角形上での三元共晶点
    ax1 = fig.add_subplot(121)
    
    # 三角形の頂点
    vertices = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3)/2]])
    triangle = Polygon(vertices, fill=False, edgecolor='black', linewidth=2)
    ax1.add_patch(triangle)
    
    ax1.text(-0.05, -0.05, 'A', fontsize=14, fontweight='bold')
    ax1.text(1.05, -0.05, 'B', fontsize=14, fontweight='bold')
    ax1.text(0.5, np.sqrt(3)/2 + 0.05, 'C', fontsize=14, fontweight='bold', ha='center')
    
    # 三元共晶点
    eutectic_comp = (0.40, 0.35, 0.25)  # A, B, C
    e_x, e_y = ternary_to_cartesian(*eutectic_comp)
    ax1.plot(e_x, e_y, 'r*', markersize=20, label='三元共晶点 E')
    
    # 平衡する3つの固相の組成
    alpha_comp = (0.85, 0.10, 0.05)
    beta_comp = (0.15, 0.75, 0.10)
    gamma_comp = (0.20, 0.15, 0.65)
    
    alpha_x, alpha_y = ternary_to_cartesian(*alpha_comp)
    beta_x, beta_y = ternary_to_cartesian(*beta_comp)
    gamma_x, gamma_y = ternary_to_cartesian(*gamma_comp)
    
    ax1.plot(alpha_x, alpha_y, 'go', markersize=10, label='α相')
    ax1.plot(beta_x, beta_y, 'bo', markersize=10, label='β相')
    ax1.plot(gamma_x, gamma_y, 'mo', markersize=10, label='γ相')
    
    # タイトライアングル（tie-triangle）
    tie_triangle = Polygon(
        [ternary_to_cartesian(*alpha_comp),
         ternary_to_cartesian(*beta_comp),
         ternary_to_cartesian(*gamma_comp)],
        fill=False, edgecolor='red', linewidth=2, linestyle='--', label='タイトライアングル'
    )
    ax1.add_patch(tie_triangle)
    
    # 組成ラベル
    ax1.text(alpha_x + 0.05, alpha_y, 'α', fontsize=11, color='green', fontweight='bold')
    ax1.text(beta_x + 0.05, beta_y, 'β', fontsize=11, color='blue', fontweight='bold')
    ax1.text(gamma_x + 0.05, gamma_y, 'γ', fontsize=11, color='purple', fontweight='bold')
    ax1.text(e_x + 0.03, e_y + 0.05, 'E', fontsize=11, color='red', fontweight='bold')
    
    ax1.set_xlim(-0.1, 1.1)
    ax1.set_ylim(-0.1, 1.0)
    ax1.set_aspect('equal')
    ax1.axis('off')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.set_title('(a) 三元共晶点とタイトライアングル', fontsize=13, fontweight='bold')
    
    # 右図: 冷却曲線
    ax2 = fig.add_subplot(122)
    
    time = np.linspace(0, 100, 500)
    
    # 冷却曲線（三元共晶組成）
    temp_eutectic = 900 - 5*time
    temp_eutectic[temp_eutectic < 550] = 550  # 共晶温度で停留
    temp_eutectic[time > 60] = 550 - 3*(time[time > 60] - 60)
    
    # 冷却曲線（非共晶組成）
    temp_noneutectic = 950 - 5*time
    temp_noneutectic[(temp_noneutectic < 600) & (temp_noneutectic > 550)] = \
        600 - 0.5*(time[(temp_noneutectic < 600) & (temp_noneutectic > 550)] - 50)
    temp_noneutectic[temp_noneutectic < 550] = 550
    temp_noneutectic[time > 70] = 550 - 3*(time[time > 70] - 70)
    
    ax2.plot(time, temp_eutectic, 'r-', linewidth=2, label='共晶組成 (E点)')
    ax2.plot(time, temp_noneutectic, 'b-', linewidth=2, label='非共晶組成')
    
    # 共晶温度の線
    ax2.axhline(550, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    ax2.text(10, 560, '三元共晶温度 T_E', fontsize=10, color='gray')
    
    ax2.set_xlabel('時間 (任意単位)', fontsize=12)
    ax2.set_ylabel('温度 (℃)', fontsize=12)
    ax2.set_title('(b) 冷却曲線: 共晶温度での停留', fontsize=13, fontweight='bold')
    ax2.set_xlim(0, 100)
    ax2.set_ylim(400, 1000)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('ternary_eutectic.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("📌 三元共晶反応の特徴:")
    print(f"・ 共晶組成: A={eutectic_comp[0]*100:.0f}%, B={eutectic_comp[1]*100:.0f}%, C={eutectic_comp[2]*100:.0f}%")
    print("・ 反応: L → α + β + γ")
    print("・ 冷却曲線: 共晶温度で顕著な停留（潜熱の放出）")
    print("・ タイトライアングル: 3つの固相の平衡組成を結ぶ三角形")

#### ギブスの相律と三元系

ギブスの相律 \\( F = C - P + 2 \\) を三元系に適用すると：

  * **成分数** \\( C = 3 \\)
  * **単相領域（P=1）** : \\( F = 4 \\) （温度、圧力、組成2変数）
  * **二相領域（P=2）** : \\( F = 3 \\) （等温・等圧下で組成1変数が自由）
  * **三相領域（P=3）** : \\( F = 2 \\) （等温・等圧下で自由度0 → 組成固定）
  * **四相共存（P=4）** : \\( F = 1 \\) （不変点: 温度と組成が固定）

三元共晶反応では、液相Lと3つの固相α, β, γが共存（P=4）するため、特定の温度と組成でのみ起こります。

## 6\. CALPHAD法の原理

**CALPHAD法** （CALculation of PHAse Diagrams）は、熱力学データベースを用いて相図を計算する手法です。実験データと理論モデルを組み合わせて、複雑な多元系の相平衡を予測します。

### 6.1 CALPHAD法の基本概念

  * **ギブス自由エネルギー最小化** : 平衡状態では系全体のギブス自由エネルギーGが最小
  * **相モデル** : 各相（固相、液相、化合物など）のギブス自由エネルギーを組成と温度の関数として表現
  * **熱力学データベース** : 実験データから最適化されたモデルパラメータを格納
  * **相平衡計算** : 全ての相のGを計算し、最小となる相の組み合わせを決定

    
    
    ```mermaid
    graph TD
        A[実験データ] --> B[熱力学モデリング]
        B --> C[パラメータ最適化]
        C --> D[熱力学データベースTDB file]
        D --> E[相図計算エンジンThermo-Calc, pycalphad]
        E --> F[相図出力]
        E --> G[熱力学量計算]
    
        H[新規材料設計] --> E
        I[プロセスシミュレーション] --> E
    
        style D fill:#f093fb,stroke:#f5576c,color:#fff
        style E fill:#f093fb,stroke:#f5576c,color:#fff
    ```

#### CALPHAD法の優位性

  1. **外挿能力** : 実験データがない領域（高温、極端な組成）の相図を予測
  2. **多元系への拡張** : 二元系データから三元系、四元系以上へ外挿可能
  3. **時間とコストの削減** : 実験の試行回数を大幅に削減
  4. **統合的アプローチ** : 相図だけでなく、熱容量、活量、化学ポテンシャルなども計算可能

### 6.2 相のギブス自由エネルギーモデル

二元系A-Bの溶体相のギブス自由エネルギーは、以下の形式で表現されます：

\\[ G_m = x_A {}^0G_A + x_B {}^0G_B + RT(x_A \ln x_A + x_B \ln x_B) + {}^{\text{ex}}G_m \\]

  * \\( {}^0G_A, {}^0G_B \\): 純成分AとBのギブス自由エネルギー（標準状態）
  * \\( RT(x_A \ln x_A + x_B \ln x_B) \\): 理想溶体の混合エントロピー項
  * \\( {}^{\text{ex}}G_m \\): 過剰ギブス自由エネルギー（非理想性を表現）

過剰ギブス自由エネルギーは、**Redlich-Kister多項式** で近似されます：

\\[ {}^{\text{ex}}G_m = x_A x_B \sum_{i=0}^{n} {}^iL_{A,B} (x_A - x_B)^i \\]

  * \\( {}^iL_{A,B} \\): 相互作用パラメータ（温度依存: \\( L = a + bT + cT\ln T + \cdots \\)）
  * \\( i \\): 多項式の次数（通常0〜2次）

#### コード例6: Redlich-Kister式による三元系モデリング
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    
    # Redlich-Kister多項式の実装
    def redlich_kister_binary(x_A, L0, L1=0, L2=0):
        """二元系A-BのRedlich-Kister過剰ギブス自由エネルギー"""
        x_B = 1 - x_A
        ex_G = x_A * x_B * (L0 + L1*(x_A - x_B) + L2*(x_A - x_B)**2)
        return ex_G
    
    def gibbs_binary(x_A, G0_A, G0_B, T, L0, L1=0, L2=0):
        """二元系のギブス自由エネルギー"""
        R = 8.314  # J/(mol·K)
        x_B = 1 - x_A
    
        # ゼロ除算を回避
        x_A = np.clip(x_A, 1e-10, 1-1e-10)
        x_B = np.clip(x_B, 1e-10, 1-1e-10)
    
        # 理想混合項
        G_ideal = x_A * G0_A + x_B * G0_B
        G_mix = R * T * (x_A * np.log(x_A) + x_B * np.log(x_B))
    
        # 過剰項
        G_ex = redlich_kister_binary(x_A, L0, L1, L2)
    
        return G_ideal + G_mix + G_ex
    
    # 二元系のギブス自由エネルギー曲線
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 左図: 理想溶体 vs 非理想溶体
    ax1 = axes[0]
    x = np.linspace(0.001, 0.999, 200)
    T = 1000  # K
    R = 8.314
    
    G0_A = 0      # 純A
    G0_B = 5000   # 純B（5 kJ/mol高い）
    
    # 理想溶体
    G_ideal = x * G0_A + (1-x) * G0_B + R*T*(x*np.log(x) + (1-x)*np.log(1-x))
    
    # 非理想溶体（正の偏倚 → 相分離傾向）
    L0_positive = 15000  # J/mol
    G_positive = gibbs_binary(x, G0_A, G0_B, T, L0_positive)
    
    # 非理想溶体（負の偏倚 → 化合物形成傾向）
    L0_negative = -10000  # J/mol
    G_negative = gibbs_binary(x, G0_A, G0_B, T, L0_negative)
    
    ax1.plot(x*100, G_ideal/1000, 'k-', linewidth=2, label='理想溶体 (L₀=0)')
    ax1.plot(x*100, G_positive/1000, 'r-', linewidth=2, label=f'正の偏倚 (L₀={L0_positive/1000:.0f} kJ/mol)')
    ax1.plot(x*100, G_negative/1000, 'b-', linewidth=2, label=f'負の偏倚 (L₀={L0_negative/1000:.0f} kJ/mol)')
    
    ax1.set_xlabel('B含有量 (at%)', fontsize=12)
    ax1.set_ylabel('ギブス自由エネルギー (kJ/mol)', fontsize=12)
    ax1.set_title('(a) Redlich-Kister式: 理想溶体からの偏倚', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 100)
    
    # 右図: 温度依存性
    ax2 = axes[1]
    temperatures = [800, 1000, 1200, 1400]  # K
    L0 = 15000  # J/mol
    
    for T in temperatures:
        G = gibbs_binary(x, G0_A, G0_B, T, L0)
        ax2.plot(x*100, G/1000, linewidth=2, label=f'T = {T} K')
    
    ax2.set_xlabel('B含有量 (at%)', fontsize=12)
    ax2.set_ylabel('ギブス自由エネルギー (kJ/mol)', fontsize=12)
    ax2.set_title('(b) ギブス自由エネルギーの温度依存性', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 100)
    
    plt.tight_layout()
    plt.savefig('redlich_kister.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("📌 Redlich-Kister式の解釈:")
    print("・ L₀ > 0: 正の偏倚 → A-B間の相互作用が弱い → 相分離傾向")
    print("・ L₀ < 0: 負の偏倚 → A-B間の相互作用が強い → 化合物形成傾向")
    print("・ 温度上昇: 混合エントロピー項（RT ln x）の寄与が増加 → 混合が有利に")

#### 💡 三元系へのモデル拡張

三元系A-B-Cでは、二元系のパラメータ（A-B、B-C、C-A）に加えて、**三元相互作用パラメータ** を導入します：

\\[ {}^{\text{ex}}G_m^{\text{ABC}} = {}^{\text{ex}}G_m^{\text{AB}} + {}^{\text{ex}}G_m^{\text{BC}} + {}^{\text{ex}}G_m^{\text{CA}} + x_A x_B x_C L_{\text{ABC}} \\]

ここで、\\( L_{\text{ABC}} \\)は三元相互作用パラメータです。多くの場合、二元系データからの外挿で十分な精度が得られるため、\\( L_{\text{ABC}} = 0 \\)と近似されます。

## 7\. CALPHAD法のワークフロー

CALPHAD法による相図計算は、以下のステップで実行されます：

#### CALPHADワークフローの5ステップ

  1. **文献調査** : 既存の実験データ（相図、熱容量、活量など）を収集
  2. **モデル選択** : 各相（液相、固溶体、化合物）の適切な熱力学モデルを選択
  3. **パラメータ最適化** : 実験データに最もフィットするモデルパラメータを最適化
  4. **データベース構築** : 最適化されたパラメータをTDBファイル（Thermo-Calc DataBase）に格納
  5. **相図計算と検証** : データベースを用いて相図を計算し、実験データと比較検証

### 7.1 熱力学データベースの構造

CALPHAD法では、**TDBファイル** （Thermo-Calc DataBase format）に熱力学データが格納されます。代表的なデータベース：

  * **SGTE（Scientific Group Thermodata Europe）** : 純物質データベース
  * **SSUB（SGTE Substance Database）** : 1500以上の純物質の熱力学データ
  * **TCFE（Thermo-Calc Steel and Fe-alloys Database）** : 鉄鋼材料データベース
  * **TCAL（Thermo-Calc Al-alloys Database）** : アルミニウム合金データベース

#### コード例7: CALPHADワークフローのシミュレーション（簡易版）
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.optimize import minimize
    
    # 簡易版CALPHADワークフロー: パラメータ最適化のデモンストレーション
    
    # ステップ1: 実験データ（模擬）
    # Cu-Ni系の液相線と固相線の実験データ
    exp_compositions = np.array([0, 20, 40, 60, 80, 100])  # wt% Ni
    exp_liquidus = np.array([1085, 1160, 1260, 1350, 1410, 1455])  # ℃
    exp_solidus = np.array([1085, 1130, 1220, 1310, 1390, 1455])   # ℃
    
    # ステップ2: 熱力学モデル（簡易版: Redlich-Kisterモデル）
    def calculate_phase_diagram(L0_liquid, L0_solid):
        """
        二元系相図の計算（極めて簡略化したモデル）
        実際のCALPHADではより複雑な計算が必要
        """
        compositions = np.linspace(0, 100, 100)
    
        # 液相線と固相線の簡易モデル
        liquidus = 1085 + (1455-1085)*compositions/100 + L0_liquid*compositions*(100-compositions)/10000
        solidus = 1085 + (1455-1085)*compositions/100 + L0_solid*compositions*(100-compositions)/10000
    
        return compositions, liquidus, solidus
    
    # ステップ3: パラメータ最適化
    def objective_function(params):
        """目的関数: 実験データとの誤差を最小化"""
        L0_liquid, L0_solid = params
    
        comp_calc, liq_calc, sol_calc = calculate_phase_diagram(L0_liquid, L0_solid)
    
        # 実験データ点での計算値を補間
        liq_interp = np.interp(exp_compositions, comp_calc, liq_calc)
        sol_interp = np.interp(exp_compositions, comp_calc, sol_calc)
    
        # 誤差の二乗和
        error = np.sum((liq_interp - exp_liquidus)**2) + np.sum((sol_interp - exp_solidus)**2)
    
        return error
    
    # 初期推定値
    initial_params = [0.0, 0.0]
    
    # 最適化実行
    print("🔧 パラメータ最適化中...")
    result = minimize(objective_function, initial_params, method='Nelder-Mead')
    optimal_L0_liquid, optimal_L0_solid = result.x
    
    print(f"✅ 最適化完了!")
    print(f"   最適パラメータ: L0_liquid = {optimal_L0_liquid:.2f}, L0_solid = {optimal_L0_solid:.2f}")
    print(f"   誤差: {result.fun:.2f} K²")
    
    # ステップ4: 相図計算と可視化
    comp_calc, liq_calc, sol_calc = calculate_phase_diagram(optimal_L0_liquid, optimal_L0_solid)
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # 計算された相図
    ax.plot(comp_calc, liq_calc, 'b-', linewidth=2, label='液相線（計算）')
    ax.plot(comp_calc, sol_calc, 'r-', linewidth=2, label='固相線（計算）')
    
    # 実験データ
    ax.plot(exp_compositions, exp_liquidus, 'bo', markersize=8, label='液相線（実験）')
    ax.plot(exp_compositions, exp_solidus, 'ro', markersize=8, label='固相線（実験）')
    
    # 相領域の塗りつぶし
    ax.fill_between(comp_calc, liq_calc, 1500, alpha=0.2, color='lightblue', label='L（液相）')
    ax.fill_between(comp_calc, sol_calc, liq_calc, alpha=0.2, color='lightgreen', label='L + α（二相）')
    ax.fill_between(comp_calc, 1050, sol_calc, alpha=0.2, color='lightyellow', label='α（固相）')
    
    ax.set_xlabel('Ni含有量 (wt%)', fontsize=13)
    ax.set_ylabel('温度 (℃)', fontsize=13)
    ax.set_title('CALPHADワークフロー: Cu-Ni系相図の計算（簡易モデル）', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 100)
    ax.set_ylim(1050, 1500)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('calphad_workflow.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # ステップ5: データベース出力（TDBファイル形式のイメージ）
    print("\n📄 TDBファイル出力（簡易版）:")
    print("=" * 50)
    print("$ Cu-Ni system optimized parameters")
    print("$ Database: DEMO_CU_NI")
    print("$ Date: 2025-10-27")
    print("$")
    print("ELEMENT Cu  FCC    63.546   5004.0  33.15  !")
    print("ELEMENT Ni  FCC    58.69    6536.0  29.87  !")
    print("$")
    print("PHASE LIQUID % 1 1.0 !")
    print(f"PARAMETER G(LIQUID,Cu;0)  298.15  +12964.7-9.511*T !")
    print(f"PARAMETER G(LIQUID,Ni;0)  298.15  +16414.7-9.397*T !")
    print(f"PARAMETER L(LIQUID,Cu,Ni;0)  298.15  {optimal_L0_liquid:.2f} !")
    print("$")
    print("PHASE FCC_A1 % 1 1.0 !")
    print(f"PARAMETER G(FCC_A1,Cu;0)  298.15  -7770.5+130.485*T !")
    print(f"PARAMETER G(FCC_A1,Ni;0)  298.15  -5179.2+117.854*T !")
    print(f"PARAMETER L(FCC_A1,Cu,Ni;0)  298.15  {optimal_L0_solid:.2f} !")
    print("=" * 50)

#### 💡 実際のCALPHAD計算ソフトウェア

上記は教育目的の極めて簡略化した例です。実際のCALPHAD計算では、以下のようなソフトウェアが使用されます：

  * **Thermo-Calc** : 商用、最も広く使われている（産業・学術）
  * **FactSage** : 商用、高温プロセスに強い
  * **Pandat** : 商用、相変態シミュレーションに特化
  * **pycalphad** : オープンソース（Python）、次章で詳しく学習

次章では、pycalphadを用いた実践的な相図計算を行います。

## 演習問題

#### 演習1: ギブスの三角形での組成読み取り

**問題:** Fe-Cr-Ni三元系で、ギブスの三角形上の点P（Fe: 70%, Cr: 20%, Ni: 10%）をプロットし、SUS304（Fe: 74%, Cr: 18%, Ni: 8%）との組成距離を計算せよ。

ヒント

三角座標をデカルト座標に変換してから、ユークリッド距離を計算します。組成空間での距離は、実際の材料特性の類似度を示す指標になります。

解答例
    
    
    # ギブスの三角形上での組成距離計算
    def ternary_to_cartesian(a, b, c):
        x = 0.5 * (2*b + c) / (a + b + c)
        y = (np.sqrt(3)/2) * c / (a + b + c)
        return x, y
    
    # 組成1: 点P
    P_comp = (0.70, 0.20, 0.10)
    P_x, P_y = ternary_to_cartesian(*P_comp)
    
    # 組成2: SUS304
    SUS304_comp = (0.74, 0.18, 0.08)
    SUS304_x, SUS304_y = ternary_to_cartesian(*SUS304_comp)
    
    # ユークリッド距離
    distance = np.sqrt((P_x - SUS304_x)**2 + (P_y - SUS304_y)**2)
    
    print(f"点P: ({P_x:.4f}, {P_y:.4f})")
    print(f"SUS304: ({SUS304_x:.4f}, {SUS304_y:.4f})")
    print(f"組成距離: {distance:.4f}（三角形上での規格化距離）")
    print(f"組成差: ΔFe={abs(0.70-0.74)*100:.1f}%, ΔCr={abs(0.20-0.18)*100:.1f}%, ΔNi={abs(0.10-0.08)*100:.1f}%")

#### 演習2: 等温断面図でのタイライン

**問題:** 1200℃のFe-Cr-Ni系等温断面図で、組成Fe: 60%, Cr: 25%, Ni: 15%の合金がL+γ二相領域にあるとします。液相の組成がFe: 50%, Cr: 30%, Ni: 20%、γ相の組成がFe: 65%, Cr: 22%, Ni: 13%のとき、各相の分率を求めよ。

ヒント

三角座標でのレバールールを適用します。合金組成点から各相への距離の逆比が相分率になります。

解答例
    
    
    # 三角座標でのレバールール
    alloy_comp = np.array([0.60, 0.25, 0.15])  # Fe, Cr, Ni
    L_comp = np.array([0.50, 0.30, 0.20])
    gamma_comp = np.array([0.65, 0.22, 0.13])
    
    # ベクトル距離計算
    dist_alloy_to_L = np.linalg.norm(alloy_comp - L_comp)
    dist_alloy_to_gamma = np.linalg.norm(alloy_comp - gamma_comp)
    dist_L_to_gamma = np.linalg.norm(L_comp - gamma_comp)
    
    # レバールール
    f_gamma = dist_alloy_to_L / dist_L_to_gamma
    f_L = dist_alloy_to_gamma / dist_L_to_gamma
    
    print(f"液相（L）の分率: {f_L*100:.1f}%")
    print(f"γ相の分率: {f_gamma*100:.1f}%")
    print(f"合計: {(f_L + f_gamma)*100:.1f}%")
    
    # 検証: 質量保存則
    reconstructed_comp = f_L * L_comp + f_gamma * gamma_comp
    print(f"\n検証（質量保存則）:")
    print(f"元の組成: Fe={alloy_comp[0]*100:.1f}%, Cr={alloy_comp[1]*100:.1f}%, Ni={alloy_comp[2]*100:.1f}%")
    print(f"再構成組成: Fe={reconstructed_comp[0]*100:.1f}%, Cr={reconstructed_comp[1]*100:.1f}%, Ni={reconstructed_comp[2]*100:.1f}%")

#### 演習3: Redlich-Kister式のフィッティング

**問題:** ある二元系A-Bの活量データ（1000 Kで、x_B = 0.2, 0.4, 0.6, 0.8での過剰ギブス自由エネルギーが5000, 8000, 8000, 5000 J/mol）が得られています。Redlich-Kisterパラメータ L0, L1を最小二乗法で決定せよ。

ヒント

Redlich-Kister式 \\( {}^{\text{ex}}G_m = x_A x_B (L_0 + L_1(x_A - x_B)) \\) を使い、実験データにフィットさせます。scipy.optimize.curve_fitが便利です。

解答例
    
    
    from scipy.optimize import curve_fit
    
    # 実験データ
    x_B_data = np.array([0.2, 0.4, 0.6, 0.8])
    ex_G_data = np.array([5000, 8000, 8000, 5000])  # J/mol
    
    # Redlich-Kisterモデル（L0, L1）
    def redlich_kister_model(x_B, L0, L1):
        x_A = 1 - x_B
        return x_A * x_B * (L0 + L1 * (x_A - x_B))
    
    # フィッティング
    popt, pcov = curve_fit(redlich_kister_model, x_B_data, ex_G_data)
    L0_fit, L1_fit = popt
    
    print(f"最適化されたパラメータ:")
    print(f"  L0 = {L0_fit:.1f} J/mol")
    print(f"  L1 = {L1_fit:.1f} J/mol")
    
    # 可視化
    x_B_fine = np.linspace(0.01, 0.99, 100)
    ex_G_fit = redlich_kister_model(x_B_fine, L0_fit, L1_fit)
    
    plt.figure(figsize=(9, 6))
    plt.plot(x_B_data, ex_G_data, 'ro', markersize=10, label='実験データ')
    plt.plot(x_B_fine, ex_G_fit, 'b-', linewidth=2, label=f'フィット (L₀={L0_fit:.0f}, L₁={L1_fit:.0f})')
    plt.xlabel('x_B', fontsize=13)
    plt.ylabel('過剰ギブス自由エネルギー (J/mol)', fontsize=13)
    plt.title('Redlich-Kisterパラメータの最適化', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

#### 演習4: CALPHAD法の応用

**問題:** Fe-Cr-Ni三元系で、SUS316L（Fe-17Cr-12Ni-2.5Mo）の組成を考えます。Mo添加により凝固温度がどのように変化するか、CALPHADの観点から考察せよ。（定性的な議論でよい）

ヒント

Moは高融点元素（2623℃）であり、鉄鋼中では固溶強化元素として作用します。液相の安定性への影響を考えましょう。

解答例

**定性的考察:**

  * **Mo添加の効果** : Moは高融点（2623℃）で、Feとの相互作用パラメータは正（相分離傾向）。
  * **液相線への影響** : Mo添加により液相線温度は**上昇** する傾向（凝固温度範囲が広がる）。
  * **凝固偏析** : Moは凝固時に液相に濃化しやすい（分配係数 k < 1）ため、凝固の最終段階でMo濃度が高くなる。
  * **実用的意義** : 凝固温度範囲が広がると、溶接時の凝固割れリスクが増加。CALPHADで凝固経路を予測し、Mo量を最適化することが重要。

**CALPHAD計算（概念）** : Thermo-CalcのTCFE（鉄鋼）データベースを用いて、Fe-17Cr-12Ni-xMo（x = 0, 1, 2, 3 wt%）の垂直断面図を計算し、液相線・固相線の変化を定量化できます。

## まとめ

この章では、三元系相図の読み方とCALPHAD法の原理を学びました。

#### 重要ポイントの復習

  1. **ギブスの三角形** : 三元系の組成を2次元の正三角形上で表現。各頂点が純成分、各辺が二元系。
  2. **アイソサーマルセクション** : 特定温度での相平衡を示す。タイラインとタイトライアングルで二相・三相領域を表現。
  3. **液相面投影図** : 液相線温度の等高線図。初晶線と共晶谷が重要。冷却経路の追跡に有用。
  4. **垂直断面図** : 特定の組成比を固定した温度-組成図。二元系相図と同様の形式で解析可能。
  5. **三元共晶反応** : L → α + β + γ の不変反応。タイトライアングルで3相の平衡組成を表現。
  6. **CALPHAD法** : ギブス自由エネルギー最小化による相図計算。熱力学データベース（TDB）を活用。
  7. **Redlich-Kister式** : 溶体相の過剰ギブス自由エネルギーをモデル化。相互作用パラメータLで非理想性を表現。
  8. **CALPHADワークフロー** : 文献調査 → モデル選択 → パラメータ最適化 → データベース構築 → 相図計算・検証。

#### 💡 次のステップ

次章では、**pycalphadによる実践的な相図計算** を学びます。オープンソースのPythonライブラリpycalphadを用いて、実際の熱力学データベース（TDB）から相図を計算し、温度-組成-相分率の関係を定量的に解析します。Fe-C系、Al-Cu系、Ni基超合金など、実用材料の相図計算を通じて、CALPHAD法の強力さを体験しましょう。

#### 学習の確認

以下の質問に答えられるか確認しましょう：

  * ギブスの三角形上で、Fe-18Cr-8Niの組成点をプロットし、組成を正確に読み取れますか？
  * 1200℃の等温断面図で、L+γ二相領域内の合金の各相分率をレバールールで計算できますか？
  * 液相面投影図から、特定組成の合金の凝固開始温度と冷却経路を追跡できますか？
  * 垂直断面図を用いて、Cr/Ni比を固定した場合のFe含有量の影響を解析できますか？
  * 三元共晶反応の特徴と、タイトライアングルの意味を説明できますか？
  * Redlich-Kister式で、正の偏倚と負の偏倚がどのような物理現象に対応するか説明できますか？
  * CALPHAD法のパラメータ最適化プロセスを、実験データとの比較を含めて説明できますか？
  * TDBファイルの構造と、熱力学データベースの役割を説明できますか？

← 第4章: 二元系相図の読み方と解析（準備中） [第6章: pycalphadによる相図計算実践 →](<chapter-6.html>)

### 免責事項

  * 本コンテンツは教育・研究・情報提供のみを目的としており、専門的な助言(法律・会計・技術的保証など)を提供するものではありません。
  * 本コンテンツおよび付随するCode examplesは「現状有姿(AS IS)」で提供され、明示または黙示を問わず、商品性、特定目的適合性、権利非侵害、正確性・完全性、動作・安全性等いかなる保証もしません。
  * 外部リンク、第三者が提供するデータ・ツール・ライブラリ等の内容・可用性・安全性について、作成者および東北大学は一切の責任を負いません。
  * 本コンテンツの利用・実行・解釈により直接的・間接的・付随的・特別・結果的・懲罰的損害が生じた場合でも、適用法で許容される最大限の範囲で、作成者および東北大学は責任を負いません。
  * 本コンテンツの内容は、予告なく変更・更新・提供停止されることがあります。
  * 本コンテンツの著作権・ライセンスは明記された条件(例: CC BY 4.0)に従います。当該ライセンスは通常、無保証条項を含みます。
