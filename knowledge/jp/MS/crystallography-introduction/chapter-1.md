---
title: 第1章：結晶学の基礎と格子の概念
chapter_title: 第1章：結晶学の基礎と格子の概念
subtitle: 原子配列の規則性と美しさを理解する
reading_time: 26-32分
difficulty: 入門
code_examples: 8
---

結晶学は、物質の原子配列の規則性を研究する学問です。この章では、結晶と非晶質の違い、格子点と単位格子の概念、そして7つの結晶系の特徴を学びます。Pythonで格子構造を可視化しながら、結晶学の美しい世界へ踏み出しましょう。 

## 学習目標

この章を読むことで、以下を習得できます：

  * ✅ 結晶と非晶質の違いを理解する
  * ✅ 格子点、単位格子、格子定数の概念を習得する
  * ✅ 7つの結晶系の特徴を理解する
  * ✅ 格子定数(a, b, c, α, β, γ)の意味を理解する
  * ✅ 基本的な格子構造をPythonで可視化できる

* * *

## 1.1 結晶学とは何か

### 結晶学の定義と重要性

**結晶学（Crystallography）** は、結晶の原子配列、対称性、構造を研究する学問分野です。材料科学、化学、物理学、鉱物学、生物学など、多くの分野で基盤となる知識です。

> **結晶学** とは、結晶性物質の内部構造、対称性、および物理的・化学的性質を研究する学問である。 

**結晶学が重要な理由** :

  * **材料特性の理解** : 原子配列が機械的・電気的・光学的性質を決定する
  * **材料設計** : 望ましい特性を持つ材料を設計するための基盤
  * **X線回折解析** : 結晶構造を実験的に決定する技術の基礎
  * **新材料開発** : 半導体、触媒、医薬品など、様々な分野で応用

### 結晶と非晶質の違い

物質の原子配列には、大きく分けて2つのパターンがあります：

#### 1\. 結晶（Crystal）

**定義** : 原子や分子が**規則的に配列** している物質

**特徴** :

  * **長距離秩序** : 原子配列の規則性が遠距離まで続く
  * **周期性** : 同じパターンが繰り返される
  * **異方性** : 方向によって性質が異なる（例：劈開、屈折率）
  * **明瞭な融点** : 一定の温度で固体→液体に変化
  * **鋭いX線回折ピーク** : 規則的な構造により特定角度で強く回折

**代表例** :

  * **塩化ナトリウム（NaCl）** : 食塩、立方晶系
  * **シリコン（Si）** : 半導体、ダイヤモンド構造
  * **鉄（Fe）** : 構造材料、体心立方格子
  * **水晶（SiO₂）** : 六方晶系

#### 2\. 非晶質・アモルファス（Amorphous）

**定義** : 原子や分子が**不規則に配列** している物質

**特徴** :

  * **短距離秩序のみ** : 近距離では規則性があるが、遠距離では無秩序
  * **周期性なし** : 繰り返しパターンが存在しない
  * **等方性** : 方向によって性質が同じ
  * **ガラス転移温度** : 徐々に軟化する（明確な融点なし）
  * **ブロードなX線回折パターン** : 鋭いピークが現れない

**代表例** :

  * **ガラス（SiO₂ガラス）** : 窓ガラス、光ファイバー
  * **アモルファスシリコン（a-Si）** : 太陽電池、薄膜トランジスタ
  * **高分子ガラス** : ポリスチレン、PMMA
  * **金属ガラス** : 急冷凝固により作製される合金

#### 結晶と非晶質の比較表

特性 | 結晶 | 非晶質（アモルファス）  
---|---|---  
**原子配列** | 規則的（長距離秩序） | 不規則（短距離秩序のみ）  
**周期性** | あり | なし  
**対称性** | 高い（点群・空間群） | 低い（等方的）  
**異方性** | あり（方向依存） | なし（等方的）  
**融解** | 明瞭な融点 | ガラス転移（徐々に軟化）  
**X線回折** | 鋭いピーク（Bragg反射） | ブロードなハロー  
**密度** | 高い（充填率が高い） | やや低い（空隙が多い）  
**熱力学的安定性** | 安定（自由エネルギー最小） | 準安定（結晶化しうる）  
  
### 周期性と対称性の概念

結晶構造を理解する上で、2つの重要な概念があります：

#### 1\. 周期性（Periodicity）

**定義** : 同じパターンが一定の間隔で繰り返されること

結晶では、原子配列が3次元空間で周期的に繰り返されます。この繰り返しの最小単位を**単位格子（Unit Cell）** と呼びます。

**数学的表現** :

位置 $\vec{r}$ に原子があるとき、以下の位置にも同じ原子配列が存在します：

$$\vec{r}' = \vec{r} + n_1\vec{a} + n_2\vec{b} + n_3\vec{c}$$

ここで、$\vec{a}, \vec{b}, \vec{c}$ は格子ベクトル、$n_1, n_2, n_3$ は整数です。

#### 2\. 対称性（Symmetry）

**定義** : ある操作を行っても元の状態と区別できない性質

**対称操作の種類** :

  * **並進対称性（Translation）** : 一定方向に平行移動
  * **回転対称性（Rotation）** : ある軸周りに回転（1回、2回、3回、4回、6回対称）
  * **鏡映対称性（Reflection）** : 鏡面に対して反転
  * **反転対称性（Inversion）** : ある点に対して反転

**重要な制約** : 結晶では、周期性と両立する回転対称性は**1回、2回、3回、4回、6回のみ** です（5回対称は存在しません）。

* * *

## 1.2 格子点と単位格子

### 格子点（Lattice Point）

**定義** : 周期的に繰り返される構造の基準点

格子点は、原子そのものではなく、**原子配列が同一な位置を示す抽象的な点** です。すべての格子点から見た周囲の環境は同一です。

**重要な性質** :

  * すべての格子点は等価である
  * 格子点から見た原子配列は、すべて同じである
  * 格子点の配列だけでは、原子の種類や位置はわからない

### 単位格子（Unit Cell）

**定義** : 結晶構造を表現する最小の繰り返し単位

単位格子を3次元空間に繰り返し並べることで、無限に広がる結晶構造を構築できます。

**単位格子の選び方** :

  * **プリミティブ格子（Primitive Cell）** : 格子点が頂点のみにある最小単位（1個の格子点を含む）
  * **慣用単位格子（Conventional Cell）** : 対称性を表現しやすい単位（複数の格子点を含むこともある）

### 格子定数（Lattice Parameters）

単位格子の形状は、**6つの格子定数** で完全に記述されます：

#### 長さのパラメータ（3つ）

  * **$a$** : 単位格子のx方向の長さ（単位: Å = 10⁻¹⁰ m）
  * **$b$** : 単位格子のy方向の長さ
  * **$c$** : 単位格子のz方向の長さ

#### 角度のパラメータ（3つ）

  * **$\alpha$** : $b$ と $c$ のなす角度（単位: °）
  * **$\beta$** : $c$ と $a$ のなす角度
  * **$\gamma$** : $a$ と $b$ のなす角度

**単位格子の体積** :

格子定数から、単位格子の体積 $V$ が計算できます：

$$V = abc\sqrt{1 - \cos^2\alpha - \cos^2\beta - \cos^2\gamma + 2\cos\alpha\cos\beta\cos\gamma}$$

特殊なケースでは、より簡単な式になります：

  * **立方晶** : $V = a^3$
  * **正方晶** : $V = a^2c$
  * **直方晶** : $V = abc$

* * *

## 1.3 7つの結晶系

すべての結晶構造は、対称性に基づいて**7つの結晶系（Crystal Systems）** に分類されます。これは、格子定数の間の関係と対称性によって決まります。
    
    
    ```mermaid
    graph TD
        A[7つの結晶系] --> B[三斜晶系Triclinic]
        A --> C[単斜晶系Monoclinic]
        A --> D[直方晶系Orthorhombic]
        A --> E[正方晶系Tetragonal]
        A --> F[六方晶系Hexagonal]
        A --> G[三方晶系Trigonal]
        A --> H[立方晶系Cubic]
    
        B --> B1[対称性最低a≠b≠c, α≠β≠γ≠90°]
        C --> C1[1つの2回軸a≠b≠c, α=γ=90°≠β]
        D --> D1[3つの垂直2回軸a≠b≠c, α=β=γ=90°]
        E --> E1[1つの4回軸a=b≠c, α=β=γ=90°]
        F --> F1[1つの6回軸a=b≠c, α=β=90°, γ=120°]
        G --> G1[1つの3回軸a=b=c, α=β=γ≠90°]
        H --> H1[対称性最高a=b=c, α=β=γ=90°]
    
        style A fill:#f093fb,stroke:#f5576c,stroke-width:2px,color:#fff
        style B fill:#fce7f3,stroke:#f093fb,stroke-width:1px
        style C fill:#fce7f3,stroke:#f093fb,stroke-width:1px
        style D fill:#fce7f3,stroke:#f093fb,stroke-width:1px
        style E fill:#fce7f3,stroke:#f093fb,stroke-width:1px
        style F fill:#fce7f3,stroke:#f093fb,stroke-width:1px
        style G fill:#fce7f3,stroke:#f093fb,stroke-width:1px
        style H fill:#fce7f3,stroke:#f093fb,stroke-width:1px
    ```

### 1\. 三斜晶系（Triclinic）

**格子定数の関係** :

$$a \neq b \neq c, \quad \alpha \neq \beta \neq \gamma \neq 90°$$

**対称性** : 最も低い（並進と反転のみ）

**代表例** :

  * 長石（Albite, NaAlSi₃O₈）
  * 硫酸銅五水和物（CuSO₄·5H₂O）

### 2\. 単斜晶系（Monoclinic）

**格子定数の関係** :

$$a \neq b \neq c, \quad \alpha = \gamma = 90° \neq \beta$$

**対称性** : 1つの2回回転軸または鏡面

**代表例** :

  * 石膏（CaSO₄·2H₂O）
  * 単斜硫黄（S）

### 3\. 直方晶系（Orthorhombic）

**格子定数の関係** :

$$a \neq b \neq c, \quad \alpha = \beta = \gamma = 90°$$

**対称性** : 3つの互いに垂直な2回回転軸

**代表例** :

  * 斜方硫黄（α-S）
  * トパーズ（Al₂SiO₄(F,OH)₂）
  * バライト（BaSO₄）

### 4\. 正方晶系（Tetragonal）

**格子定数の関係** :

$$a = b \neq c, \quad \alpha = \beta = \gamma = 90°$$

**対称性** : 1つの4回回転軸（c軸）

**代表例** :

  * ルチル型二酸化チタン（TiO₂）
  * ジルコン（ZrSiO₄）
  * 錫（β-Sn, 白色錫）

### 5\. 六方晶系（Hexagonal）

**格子定数の関係** :

$$a = b \neq c, \quad \alpha = \beta = 90°, \quad \gamma = 120°$$

**対称性** : 1つの6回回転軸（c軸）

**代表例** :

  * 水晶（α-SiO₂）
  * ベリリウム（Be）
  * マグネシウム（Mg）
  * 亜鉛（Zn）
  * グラファイト（C）

### 6\. 三方晶系（Trigonal / Rhombohedral）

**格子定数の関係** :

$$a = b = c, \quad \alpha = \beta = \gamma \neq 90°$$

**対称性** : 1つの3回回転軸

**代表例** :

  * 方解石（CaCO₃）
  * コランダム（α-Al₂O₃）
  * 水銀（Hg, 低温相）

**注意** : 三方晶系は六方晶系の格子で記述されることもあります（六方格子+3回回転軸）。

### 7\. 立方晶系（Cubic）

**格子定数の関係** :

$$a = b = c, \quad \alpha = \beta = \gamma = 90°$$

**対称性** : 最も高い（4つの3回回転軸）

**代表例** :

  * 塩化ナトリウム（NaCl）
  * ダイヤモンド（C）
  * シリコン（Si）
  * 鉄（α-Fe, フェライト）
  * 銅（Cu）
  * 金（Au）

### 7つの結晶系の比較表

結晶系 | 格子定数の関係 | 軸角の関係 | 主要対称要素 | 代表例  
---|---|---|---|---  
**三斜晶** | $a \neq b \neq c$ | $\alpha \neq \beta \neq \gamma \neq 90°$ | 反転中心のみ | 長石、CuSO₄·5H₂O  
**単斜晶** | $a \neq b \neq c$ | $\alpha = \gamma = 90° \neq \beta$ | 1つの2回軸 | 石膏、単斜硫黄  
**直方晶** | $a \neq b \neq c$ | $\alpha = \beta = \gamma = 90°$ | 3つの垂直2回軸 | 斜方硫黄、BaSO₄  
**正方晶** | $a = b \neq c$ | $\alpha = \beta = \gamma = 90°$ | 1つの4回軸 | TiO₂、β-Sn  
**六方晶** | $a = b \neq c$ | $\alpha = \beta = 90°, \gamma = 120°$ | 1つの6回軸 | 水晶、Mg、Zn  
**三方晶** | $a = b = c$ | $\alpha = \beta = \gamma \neq 90°$ | 1つの3回軸 | CaCO₃、α-Al₂O₃  
**立方晶** | $a = b = c$ | $\alpha = \beta = \gamma = 90°$ | 4つの3回軸 | NaCl、Si、Fe、Cu  
  
* * *

## 1.4 Pythonによる格子の可視化

ここから、Pythonを使って格子構造を可視化し、結晶学の概念を視覚的に理解しましょう。

### 環境準備

必要なライブラリをインストールします：
    
    
    # 必要なライブラリのインストール
    pip install numpy matplotlib pandas plotly
    

### コード例1: 7つの結晶系の格子定数テーブル作成

まず、7つの結晶系の格子定数の関係を整理したテーブルを作成します。
    
    
    import pandas as pd
    import numpy as np
    
    # 7つの結晶系の格子定数の関係を定義
    crystal_systems_data = {
        '結晶系': ['三斜晶', '単斜晶', '直方晶', '正方晶', '六方晶', '三方晶', '立方晶'],
        '英名': ['Triclinic', 'Monoclinic', 'Orthorhombic', 'Tetragonal',
                 'Hexagonal', 'Trigonal', 'Cubic'],
        '軸長の関係': ['a≠b≠c', 'a≠b≠c', 'a≠b≠c', 'a=b≠c', 'a=b≠c', 'a=b=c', 'a=b=c'],
        'α': ['≠90°', '90°', '90°', '90°', '90°', '≠90°', '90°'],
        'β': ['≠90°', '≠90°', '90°', '90°', '90°', '=α', '90°'],
        'γ': ['≠90°', '90°', '90°', '90°', '120°', '=α', '90°'],
        '主要対称要素': ['反転のみ', '1つの2回軸', '3つの2回軸', '1つの4回軸',
                        '1つの6回軸', '1つの3回軸', '4つの3回軸'],
        '代表例': ['長石', '石膏', '斜方硫黄', 'TiO₂', 'Mg', 'CaCO₃', 'NaCl']
    }
    
    # DataFrameの作成
    df_crystal_systems = pd.DataFrame(crystal_systems_data)
    
    # 表示
    print("=" * 100)
    print("7つの結晶系の格子定数まとめ")
    print("=" * 100)
    print(df_crystal_systems.to_string(index=False))
    print("=" * 100)
    
    # 具体的な格子定数の例
    print("\n具体的な格子定数の例（代表的な物質）:")
    print("-" * 80)
    
    examples = [
        {'物質': 'NaCl（塩化ナトリウム）', '結晶系': '立方晶',
         'a': 5.64, 'b': 5.64, 'c': 5.64, 'α': 90, 'β': 90, 'γ': 90},
        {'物質': 'Si（シリコン）', '結晶系': '立方晶',
         'a': 5.43, 'b': 5.43, 'c': 5.43, 'α': 90, 'β': 90, 'γ': 90},
        {'物質': 'TiO₂（ルチル）', '結晶系': '正方晶',
         'a': 4.59, 'b': 4.59, 'c': 2.96, 'α': 90, 'β': 90, 'γ': 90},
        {'物質': 'Mg（マグネシウム）', '結晶系': '六方晶',
         'a': 3.21, 'b': 3.21, 'c': 5.21, 'α': 90, 'β': 90, 'γ': 120},
        {'物質': 'α-Fe（フェライト）', '結晶系': '立方晶',
         'a': 2.87, 'b': 2.87, 'c': 2.87, 'α': 90, 'β': 90, 'γ': 90},
    ]
    
    for ex in examples:
        print(f"{ex['物質']:30s} ({ex['結晶系']})")
        print(f"  a={ex['a']:.2f}Å, b={ex['b']:.2f}Å, c={ex['c']:.2f}Å")
        print(f"  α={ex['α']}°, β={ex['β']}°, γ={ex['γ']}°")
        print()
    

**出力例** :
    
    
    ====================================================================================================
    7つの結晶系の格子定数まとめ
    ====================================================================================================
    結晶系  英名         軸長の関係  α    β    γ    主要対称要素    代表例
    三斜晶  Triclinic    a≠b≠c      ≠90° ≠90° ≠90° 反転のみ        長石
    単斜晶  Monoclinic   a≠b≠c      90°  ≠90° 90°  1つの2回軸      石膏
    直方晶  Orthorhombic a≠b≠c      90°  90°  90°  3つの2回軸      斜方硫黄
    正方晶  Tetragonal   a=b≠c      90°  90°  90°  1つの4回軸      TiO₂
    六方晶  Hexagonal    a=b≠c      90°  90°  120° 1つの6回軸      Mg
    三方晶  Trigonal     a=b=c      ≠90° =α   =α   1つの3回軸      CaCO₃
    立方晶  Cubic        a=b=c      90°  90°  90°  4つの3回軸      NaCl
    ====================================================================================================
    

**解説** : このテーブルにより、7つの結晶系の格子定数の関係が一目で理解できます。対称性が高いほど、制約が多く（例：立方晶は a=b=c）、パラメータが少なくなります。

### コード例2: 2D格子の生成と可視化（正方格子）

まず、2次元の正方格子を生成し、格子点と単位格子を可視化します。
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    # 2D正方格子のパラメータ
    a = 1.0  # 格子定数（任意単位）
    n_cells = 5  # 表示する単位格子の数（x, y方向）
    
    # 格子点の生成
    x_points = []
    y_points = []
    
    for i in range(n_cells + 1):
        for j in range(n_cells + 1):
            x_points.append(i * a)
            y_points.append(j * a)
    
    # プロット作成
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # 格子点をプロット
    ax.scatter(x_points, y_points, s=150, c='#f093fb',
               edgecolors='black', linewidths=2, zorder=3, label='格子点')
    
    # 単位格子の辺を描画
    for i in range(n_cells):
        for j in range(n_cells):
            # 単位格子の4辺を描画
            # 下辺
            ax.plot([i*a, (i+1)*a], [j*a, j*a], 'b-', linewidth=1.5, alpha=0.6)
            # 右辺
            ax.plot([(i+1)*a, (i+1)*a], [j*a, (j+1)*a], 'b-', linewidth=1.5, alpha=0.6)
            # 上辺
            ax.plot([(i+1)*a, i*a], [(j+1)*a, (j+1)*a], 'b-', linewidth=1.5, alpha=0.6)
            # 左辺
            ax.plot([i*a, i*a], [(j+1)*a, j*a], 'b-', linewidth=1.5, alpha=0.6)
    
    # 最初の単位格子を強調表示
    ax.plot([0, a], [0, 0], 'r-', linewidth=3, label='単位格子')
    ax.plot([a, a], [0, a], 'r-', linewidth=3)
    ax.plot([a, 0], [a, a], 'r-', linewidth=3)
    ax.plot([0, 0], [a, 0], 'r-', linewidth=3)
    
    # 格子ベクトルを矢印で表示
    ax.annotate('', xy=(a, 0), xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', lw=2.5, color='green'))
    ax.annotate('', xy=(0, a), xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', lw=2.5, color='green'))
    ax.text(a/2, -0.3, r'$\vec{a}$', fontsize=14, color='green', fontweight='bold')
    ax.text(-0.3, a/2, r'$\vec{b}$', fontsize=14, color='green', fontweight='bold')
    
    # 軸設定
    ax.set_xlim(-0.5, n_cells * a + 0.5)
    ax.set_ylim(-0.5, n_cells * a + 0.5)
    ax.set_aspect('equal')
    ax.set_xlabel('x', fontsize=12, fontweight='bold')
    ax.set_ylabel('y', fontsize=12, fontweight='bold')
    ax.set_title('2D正方格子の可視化', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("格子点の総数:", len(x_points))
    print(f"格子定数: a = {a}")
    print(f"単位格子の面積: {a**2:.2f}")
    print(f"表示範囲内の単位格子数: {n_cells * n_cells}")
    

**解説** : このプロットにより、格子点が周期的に配列し、単位格子（赤枠）を繰り返すことで全体の格子が構築されることが視覚的に理解できます。格子ベクトル $\vec{a}$ と $\vec{b}$ が格子の基本構造を定義します。

### コード例3: 3D単位格子の可視化（立方晶）

3次元の立方晶単位格子をPlotlyで可視化します。
    
    
    import numpy as np
    import plotly.graph_objects as go
    
    # 立方晶単位格子のパラメータ
    a = 1.0  # 格子定数
    
    # 単位格子の頂点座標（8個）
    vertices = np.array([
        [0, 0, 0],
        [a, 0, 0],
        [a, a, 0],
        [0, a, 0],
        [0, 0, a],
        [a, 0, a],
        [a, a, a],
        [0, a, a]
    ])
    
    # 単位格子の辺（12本）
    edges = [
        [0, 1], [1, 2], [2, 3], [3, 0],  # 底面
        [4, 5], [5, 6], [6, 7], [7, 4],  # 上面
        [0, 4], [1, 5], [2, 6], [3, 7]   # 縦の辺
    ]
    
    # プロット作成
    fig = go.Figure()
    
    # 辺を描画
    for edge in edges:
        v1, v2 = vertices[edge[0]], vertices[edge[1]]
        fig.add_trace(go.Scatter3d(
            x=[v1[0], v2[0]],
            y=[v1[1], v2[1]],
            z=[v1[2], v2[2]],
            mode='lines',
            line=dict(color='blue', width=6),
            showlegend=False,
            hoverinfo='skip'
        ))
    
    # 頂点（格子点）を描画
    fig.add_trace(go.Scatter3d(
        x=vertices[:, 0],
        y=vertices[:, 1],
        z=vertices[:, 2],
        mode='markers',
        marker=dict(size=10, color='#f093fb',
                    line=dict(color='black', width=2)),
        name='格子点',
        hovertemplate='座標: (%{x:.2f}, %{y:.2f}, %{z:.2f})'
    ))
    
    # 軸ラベル
    fig.update_layout(
        title='立方晶単位格子の3D可視化',
        scene=dict(
            xaxis_title='x (Å)',
            yaxis_title='y (Å)',
            zaxis_title='z (Å)',
            aspectmode='cube',
            xaxis=dict(range=[-0.2, a+0.2]),
            yaxis=dict(range=[-0.2, a+0.2]),
            zaxis=dict(range=[-0.2, a+0.2])
        ),
        width=700,
        height=700
    )
    
    fig.show()
    
    print("立方晶単位格子の情報:")
    print(f"格子定数: a = b = c = {a} Å")
    print(f"軸角: α = β = γ = 90°")
    print(f"単位格子の体積: V = a³ = {a**3:.2f} ų")
    print(f"格子点数（頂点のみ）: 8個（1個の格子点に相当: 8 × 1/8 = 1）")
    

**解説** : この3D可視化により、立方晶の単位格子が立方体形状であることが直感的に理解できます。頂点にある8個の格子点は、隣接する単位格子と共有されるため、1つの単位格子に含まれる格子点は実質1個です（8 × 1/8 = 1）。

### コード例4: 格子定数から単位格子体積を計算

格子定数から単位格子の体積を計算する関数を作成します。
    
    
    import numpy as np
    
    def calc_unit_cell_volume(a, b, c, alpha, beta, gamma):
        """
        格子定数から単位格子の体積を計算
    
        Parameters:
        -----------
        a, b, c : float
            軸長（Å）
        alpha, beta, gamma : float
            軸角（度）
    
        Returns:
        --------
        volume : float
            単位格子の体積（ų）
        """
        # 角度をラジアンに変換
        alpha_rad = np.deg2rad(alpha)
        beta_rad = np.deg2rad(beta)
        gamma_rad = np.deg2rad(gamma)
    
        # 体積の計算式
        cos_alpha = np.cos(alpha_rad)
        cos_beta = np.cos(beta_rad)
        cos_gamma = np.deg2rad(gamma)
    
        volume = a * b * c * np.sqrt(
            1 - cos_alpha**2 - cos_beta**2 - np.cos(gamma_rad)**2
            + 2 * cos_alpha * cos_beta * np.cos(gamma_rad)
        )
    
        return volume
    
    # 各結晶系の代表例で体積を計算
    examples = [
        {'名前': 'NaCl', '結晶系': '立方晶', 'a': 5.64, 'b': 5.64, 'c': 5.64,
         'alpha': 90, 'beta': 90, 'gamma': 90},
        {'名前': 'Si', '結晶系': '立方晶', 'a': 5.43, 'b': 5.43, 'c': 5.43,
         'alpha': 90, 'beta': 90, 'gamma': 90},
        {'名前': 'TiO₂', '結晶系': '正方晶', 'a': 4.59, 'b': 4.59, 'c': 2.96,
         'alpha': 90, 'beta': 90, 'gamma': 90},
        {'名前': 'Mg', '結晶系': '六方晶', 'a': 3.21, 'b': 3.21, 'c': 5.21,
         'alpha': 90, 'beta': 90, 'gamma': 120},
        {'名前': 'CaCO₃', '結晶系': '三方晶', 'a': 4.99, 'b': 4.99, 'c': 4.99,
         'alpha': 101.9, 'beta': 101.9, 'gamma': 101.9},
    ]
    
    print("=" * 70)
    print("格子定数から単位格子体積を計算")
    print("=" * 70)
    
    for ex in examples:
        volume = calc_unit_cell_volume(
            ex['a'], ex['b'], ex['c'], ex['alpha'], ex['beta'], ex['gamma']
        )
    
        print(f"\n{ex['名前']} ({ex['結晶系']})")
        print(f"  格子定数: a={ex['a']:.2f}Å, b={ex['b']:.2f}Å, c={ex['c']:.2f}Å")
        print(f"  軸角: α={ex['alpha']:.1f}°, β={ex['beta']:.1f}°, γ={ex['gamma']:.1f}°")
        print(f"  単位格子体積: V = {volume:.2f} ų")
    
    print("\n" + "=" * 70)
    
    # 立方晶の簡略式との比較
    print("\n立方晶（NaCl）の場合、簡略式 V = a³ も使えます：")
    a_nacl = 5.64
    v_simple = a_nacl ** 3
    v_general = calc_unit_cell_volume(a_nacl, a_nacl, a_nacl, 90, 90, 90)
    print(f"  簡略式: V = {a_nacl}³ = {v_simple:.2f} ų")
    print(f"  一般式: V = {v_general:.2f} ų")
    print(f"  差: {abs(v_simple - v_general):.6f} ų （誤差範囲内で一致）")
    

**出力例** :
    
    
    ======================================================================
    格子定数から単位格子体積を計算
    ======================================================================
    
    NaCl (立方晶)
      格子定数: a=5.64Å, b=5.64Å, c=5.64Å
      軸角: α=90.0°, β=90.0°, γ=90.0°
      単位格子体積: V = 179.41 ų
    
    Si (立方晶)
      格子定数: a=5.43Å, b=5.43Å, c=5.43Å
      軸角: α=90.0°, β=90.0°, γ=90.0°
      単位格子体積: V = 160.10 ų
    
    TiO₂ (正方晶)
      格子定数: a=4.59Å, b=4.59Å, c=2.96Å
      軸角: α=90.0°, β=90.0°, γ=90.0°
      単位格子体積: V = 62.35 ų
    
    Mg (六方晶)
      格子定数: a=3.21Å, b=3.21Å, c=5.21Å
      軸角: α=90.0°, β=90.0°, γ=120.0°
      単位格子体積: V = 46.49 ų
    

**解説** : この関数により、任意の結晶系の単位格子体積を一般式で計算できます。立方晶や正方晶では簡略式（$V = a^3$ や $V = a^2c$）が使えますが、一般式はすべての結晶系に適用可能です。

### コード例5: 異なる結晶系の単位格子可視化（立方晶 vs 正方晶 vs 直方晶）

3つの異なる結晶系の単位格子を並べて可視化し、違いを理解します。
    
    
    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    
    def create_unit_cell_vertices(a, b, c):
        """単位格子の頂点座標を生成"""
        return np.array([
            [0, 0, 0], [a, 0, 0], [a, b, 0], [0, b, 0],
            [0, 0, c], [a, 0, c], [a, b, c], [0, b, c]
        ])
    
    def add_unit_cell_to_fig(fig, a, b, c, row, col, title):
        """単位格子をサブプロットに追加"""
        vertices = create_unit_cell_vertices(a, b, c)
    
        edges = [
            [0, 1], [1, 2], [2, 3], [3, 0],  # 底面
            [4, 5], [5, 6], [6, 7], [7, 4],  # 上面
            [0, 4], [1, 5], [2, 6], [3, 7]   # 縦の辺
        ]
    
        # 辺を描画
        for edge in edges:
            v1, v2 = vertices[edge[0]], vertices[edge[1]]
            fig.add_trace(go.Scatter3d(
                x=[v1[0], v2[0]], y=[v1[1], v2[1]], z=[v1[2], v2[2]],
                mode='lines', line=dict(color='blue', width=4),
                showlegend=False, hoverinfo='skip'
            ), row=row, col=col)
    
        # 頂点を描画
        fig.add_trace(go.Scatter3d(
            x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
            mode='markers', marker=dict(size=6, color='#f093fb'),
            showlegend=False, hoverinfo='skip'
        ), row=row, col=col)
    
        # サブプロットのタイトルと軸設定
        scene_name = f'scene{(row-1)*3 + col}' if row > 1 or col > 1 else 'scene'
        fig.update_layout({
            scene_name: dict(
                xaxis_title='x', yaxis_title='y', zaxis_title='z',
                aspectmode='cube',
                xaxis=dict(range=[0, max(a, b, c)]),
                yaxis=dict(range=[0, max(a, b, c)]),
                zaxis=dict(range=[0, max(a, b, c)])
            )
        })
    
    # サブプロット作成（1行3列）
    fig = make_subplots(
        rows=1, cols=3,
        specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}, {'type': 'scatter3d'}]],
        subplot_titles=('立方晶 (a=b=c)', '正方晶 (a=b≠c)', '直方晶 (a≠b≠c)')
    )
    
    # 立方晶（a=b=c）
    add_unit_cell_to_fig(fig, a=2.0, b=2.0, c=2.0, row=1, col=1, title='立方晶')
    
    # 正方晶（a=b≠c）
    add_unit_cell_to_fig(fig, a=2.0, b=2.0, c=3.0, row=1, col=2, title='正方晶')
    
    # 直方晶（a≠b≠c）
    add_unit_cell_to_fig(fig, a=2.0, b=2.5, c=3.5, row=1, col=3, title='直方晶')
    
    # レイアウト調整
    fig.update_layout(
        title_text="3つの結晶系の単位格子比較",
        height=500,
        width=1200,
        showlegend=False
    )
    
    fig.show()
    
    # 各結晶系の情報を出力
    print("3つの結晶系の比較:")
    print("-" * 60)
    print("立方晶:  a = b = c = 2.0 Å, α = β = γ = 90°")
    print("         体積 V = a³ = 8.0 ų")
    print("-" * 60)
    print("正方晶:  a = b = 2.0 Å, c = 3.0 Å, α = β = γ = 90°")
    print("         体積 V = a²c = 12.0 ų")
    print("-" * 60)
    print("直方晶:  a = 2.0 Å, b = 2.5 Å, c = 3.5 Å, α = β = γ = 90°")
    print("         体積 V = abc = 17.5 ų")
    print("-" * 60)
    

**解説** : この並列比較により、立方晶→正方晶→直方晶の順に対称性が低下し、形状の制約が緩和されることが視覚的に理解できます。立方晶は最も対称性が高く、直方晶は3つの軸長がすべて異なります。

### コード例6: 格子点の生成プログラム（3D格子）

3次元空間に格子点を生成し、スーパーセル（複数の単位格子）を可視化します。
    
    
    import numpy as np
    import plotly.graph_objects as go
    
    def generate_3d_lattice_points(a, b, c, nx, ny, nz):
        """
        3D格子点を生成
    
        Parameters:
        -----------
        a, b, c : float
            格子定数（Å）
        nx, ny, nz : int
            各方向の単位格子数
    
        Returns:
        --------
        points : ndarray
            格子点座標の配列 (N, 3)
        """
        points = []
        for i in range(nx + 1):
            for j in range(ny + 1):
                for k in range(nz + 1):
                    x = i * a
                    y = j * b
                    z = k * c
                    points.append([x, y, z])
        return np.array(points)
    
    # 立方晶のスーパーセルを生成（3×3×3）
    a = 1.0  # 格子定数
    nx, ny, nz = 3, 3, 3  # 各方向の単位格子数
    
    lattice_points = generate_3d_lattice_points(a, a, a, nx, ny, nz)
    
    # プロット作成
    fig = go.Figure()
    
    # 格子点を描画
    fig.add_trace(go.Scatter3d(
        x=lattice_points[:, 0],
        y=lattice_points[:, 1],
        z=lattice_points[:, 2],
        mode='markers',
        marker=dict(size=5, color='#f093fb',
                    line=dict(color='black', width=1)),
        name='格子点',
        hovertemplate='座標: (%{x:.1f}, %{y:.1f}, %{z:.1f})'
    ))
    
    # 単位格子の辺を描画（最初の単位格子のみ）
    unit_cell_edges = [
        [[0, 0, 0], [a, 0, 0]], [[a, 0, 0], [a, a, 0]],
        [[a, a, 0], [0, a, 0]], [[0, a, 0], [0, 0, 0]],
        [[0, 0, a], [a, 0, a]], [[a, 0, a], [a, a, a]],
        [[a, a, a], [0, a, a]], [[0, a, a], [0, 0, a]],
        [[0, 0, 0], [0, 0, a]], [[a, 0, 0], [a, 0, a]],
        [[a, a, 0], [a, a, a]], [[0, a, 0], [0, a, a]]
    ]
    
    for edge in unit_cell_edges:
        fig.add_trace(go.Scatter3d(
            x=[edge[0][0], edge[1][0]],
            y=[edge[0][1], edge[1][1]],
            z=[edge[0][2], edge[1][2]],
            mode='lines',
            line=dict(color='red', width=4),
            showlegend=False,
            hoverinfo='skip'
        ))
    
    # レイアウト設定
    fig.update_layout(
        title=f'3D格子点の生成（{nx}×{ny}×{nz} スーパーセル）',
        scene=dict(
            xaxis_title='x (Å)',
            yaxis_title='y (Å)',
            zaxis_title='z (Å)',
            aspectmode='cube'
        ),
        width=700,
        height=700
    )
    
    fig.show()
    
    # 統計情報を出力
    print(f"スーパーセルの情報:")
    print(f"  格子定数: a = b = c = {a} Å")
    print(f"  単位格子数: {nx} × {ny} × {nz} = {nx*ny*nz} 個")
    print(f"  格子点総数: {len(lattice_points)} 個")
    print(f"  格子点密度: {len(lattice_points)/(nx*a*ny*a*nz*a):.2f} 個/ų")
    print(f"  全体の体積: {nx*a} × {ny*a} × {nz*a} = {nx*ny*nz*a**3:.2f} ų")
    

**解説** : このプログラムにより、任意のサイズのスーパーセル（複数の単位格子を組み合わせた大きな構造）を生成できます。格子点の周期的配列が3次元空間で繰り返されることが視覚的に理解できます。

### コード例7: スーパーセルの生成（2×2×2拡張）

単位格子を2×2×2倍に拡張したスーパーセルを生成します。
    
    
    import numpy as np
    
    def expand_unit_cell_to_supercell(unit_cell_atoms, lattice_vectors, n1, n2, n3):
        """
        単位格子をスーパーセルに拡張
    
        Parameters:
        -----------
        unit_cell_atoms : ndarray
            単位格子内の原子座標（分数座標） (N, 3)
        lattice_vectors : ndarray
            格子ベクトル (3, 3) [[a_x, a_y, a_z], [b_x, b_y, b_z], [c_x, c_y, c_z]]
        n1, n2, n3 : int
            各方向の拡張倍率
    
        Returns:
        --------
        supercell_atoms : ndarray
            スーパーセル内の原子座標（デカルト座標） (M, 3)
        """
        supercell_atoms = []
    
        for atom_frac in unit_cell_atoms:
            for i in range(n1):
                for j in range(n2):
                    for k in range(n3):
                        # 分数座標に並進を加える
                        frac_coord = atom_frac + np.array([i/n1, j/n2, k/n3])
    
                        # デカルト座標に変換
                        cart_coord = np.dot(frac_coord, lattice_vectors)
                        supercell_atoms.append(cart_coord)
    
        return np.array(supercell_atoms)
    
    # 例：単純立方格子の単位格子（格子点のみ）
    unit_cell_atoms = np.array([
        [0.0, 0.0, 0.0]  # 原点にのみ原子（分数座標）
    ])
    
    # 格子ベクトル（立方晶、a=2.0 Å）
    a = 2.0
    lattice_vectors = np.array([
        [a, 0, 0],
        [0, a, 0],
        [0, 0, a]
    ])
    
    # 2×2×2スーパーセルを生成
    supercell = expand_unit_cell_to_supercell(unit_cell_atoms, lattice_vectors, 2, 2, 2)
    
    print("2×2×2スーパーセルの生成結果:")
    print("=" * 60)
    print(f"単位格子内の原子数: {len(unit_cell_atoms)}")
    print(f"スーパーセル内の原子数: {len(supercell)} (= {len(unit_cell_atoms)} × 2³)")
    print("=" * 60)
    print("\nスーパーセル内の原子座標（デカルト座標, Å）:")
    for i, coord in enumerate(supercell, 1):
        print(f"  原子{i:2d}: ({coord[0]:5.2f}, {coord[1]:5.2f}, {coord[2]:5.2f})")
    
    # スーパーセルのサイズを計算
    supercell_size = 2 * a
    supercell_volume = supercell_size ** 3
    print(f"\nスーパーセルのサイズ: {supercell_size} × {supercell_size} × {supercell_size} Å")
    print(f"スーパーセルの体積: {supercell_volume:.2f} ų")
    print(f"原子密度: {len(supercell)/supercell_volume:.4f} 個/ų")
    

**出力例** :
    
    
    2×2×2スーパーセルの生成結果:
    ============================================================
    単位格子内の原子数: 1
    スーパーセル内の原子数: 8 (= 1 × 2³)
    ============================================================
    
    スーパーセル内の原子座標（デカルト座標, Å）:
      原子 1: ( 0.00,  0.00,  0.00)
      原子 2: ( 0.00,  0.00,  2.00)
      原子 3: ( 0.00,  2.00,  0.00)
      原子 4: ( 0.00,  2.00,  2.00)
      原子 5: ( 2.00,  0.00,  0.00)
      原子 6: ( 2.00,  0.00,  2.00)
      原子 7: ( 2.00,  2.00,  0.00)
      原子 8: ( 2.00,  2.00,  2.00)
    
    スーパーセルのサイズ: 4.0 × 4.0 × 4.0 Å
    スーパーセルの体積: 64.00 ų
    原子密度: 0.1250 個/ų
    

**解説** : スーパーセルは、計算化学やシミュレーションで頻繁に使用されます。単位格子を繰り返すことで、表面、界面、欠陥などの大規模構造をモデル化できます。

### コード例8: 結晶系判定プログラム（格子定数から判定）

格子定数を入力すると、どの結晶系に属するかを判定するプログラムを作成します。
    
    
    import numpy as np
    
    def determine_crystal_system(a, b, c, alpha, beta, gamma, tolerance=0.01):
        """
        格子定数から結晶系を判定
    
        Parameters:
        -----------
        a, b, c : float
            軸長（Å）
        alpha, beta, gamma : float
            軸角（度）
        tolerance : float
            判定の許容誤差（相対誤差）
    
        Returns:
        --------
        system : str
            結晶系の名前
        """
        # 角度の許容誤差（度）
        angle_tol = 1.0
    
        # 長さの比較（相対誤差）
        def is_equal(x, y):
            return abs(x - y) / max(x, y) < tolerance
    
        # 角度の比較
        def angle_is(angle, target):
            return abs(angle - target) < angle_tol
    
        # 立方晶: a=b=c, α=β=γ=90°
        if is_equal(a, b) and is_equal(b, c) and \
           angle_is(alpha, 90) and angle_is(beta, 90) and angle_is(gamma, 90):
            return '立方晶 (Cubic)'
    
        # 正方晶: a=b≠c, α=β=γ=90°
        elif is_equal(a, b) and not is_equal(a, c) and \
             angle_is(alpha, 90) and angle_is(beta, 90) and angle_is(gamma, 90):
            return '正方晶 (Tetragonal)'
    
        # 直方晶: a≠b≠c, α=β=γ=90°
        elif not is_equal(a, b) and not is_equal(b, c) and not is_equal(a, c) and \
             angle_is(alpha, 90) and angle_is(beta, 90) and angle_is(gamma, 90):
            return '直方晶 (Orthorhombic)'
    
        # 六方晶: a=b≠c, α=β=90°, γ=120°
        elif is_equal(a, b) and not is_equal(a, c) and \
             angle_is(alpha, 90) and angle_is(beta, 90) and angle_is(gamma, 120):
            return '六方晶 (Hexagonal)'
    
        # 三方晶: a=b=c, α=β=γ≠90°
        elif is_equal(a, b) and is_equal(b, c) and \
             is_equal(alpha, beta) and is_equal(beta, gamma) and not angle_is(alpha, 90):
            return '三方晶 (Trigonal/Rhombohedral)'
    
        # 単斜晶: a≠b≠c, α=γ=90°≠β
        elif not is_equal(a, b) and not is_equal(b, c) and not is_equal(a, c) and \
             angle_is(alpha, 90) and not angle_is(beta, 90) and angle_is(gamma, 90):
            return '単斜晶 (Monoclinic)'
    
        # 三斜晶: a≠b≠c, α≠β≠γ≠90°（最も一般的）
        else:
            return '三斜晶 (Triclinic)'
    
    # テストケース
    test_cases = [
        {'名前': 'NaCl', 'a': 5.64, 'b': 5.64, 'c': 5.64, 'alpha': 90, 'beta': 90, 'gamma': 90},
        {'名前': 'Si', 'a': 5.43, 'b': 5.43, 'c': 5.43, 'alpha': 90, 'beta': 90, 'gamma': 90},
        {'名前': 'TiO₂', 'a': 4.59, 'b': 4.59, 'c': 2.96, 'alpha': 90, 'beta': 90, 'gamma': 90},
        {'名前': 'Mg', 'a': 3.21, 'b': 3.21, 'c': 5.21, 'alpha': 90, 'beta': 90, 'gamma': 120},
        {'名前': 'CaCO₃', 'a': 4.99, 'b': 4.99, 'c': 4.99, 'alpha': 101.9, 'beta': 101.9, 'gamma': 101.9},
        {'名前': '石膏', 'a': 5.68, 'b': 15.18, 'c': 6.29, 'alpha': 90, 'beta': 113.8, 'gamma': 90},
        {'名前': '硫黄', 'a': 10.47, 'b': 12.87, 'c': 24.49, 'alpha': 90, 'beta': 90, 'gamma': 90},
    ]
    
    print("=" * 80)
    print("結晶系判定プログラム")
    print("=" * 80)
    
    for case in test_cases:
        system = determine_crystal_system(
            case['a'], case['b'], case['c'],
            case['alpha'], case['beta'], case['gamma']
        )
    
        print(f"\n{case['名前']:10s}")
        print(f"  格子定数: a={case['a']:.2f}Å, b={case['b']:.2f}Å, c={case['c']:.2f}Å")
        print(f"  軸角: α={case['alpha']:.1f}°, β={case['beta']:.1f}°, γ={case['gamma']:.1f}°")
        print(f"  → 判定結果: {system}")
    
    print("\n" + "=" * 80)
    

**出力例** :
    
    
    ================================================================================
    結晶系判定プログラム
    ================================================================================
    
    NaCl
      格子定数: a=5.64Å, b=5.64Å, c=5.64Å
      軸角: α=90.0°, β=90.0°, γ=90.0°
      → 判定結果: 立方晶 (Cubic)
    
    Si
      格子定数: a=5.43Å, b=5.43Å, c=5.43Å
      軸角: α=90.0°, β=90.0°, γ=90.0°
      → 判定結果: 立方晶 (Cubic)
    
    TiO₂
      格子定数: a=4.59Å, b=4.59Å, c=2.96Å
      軸角: α=90.0°, β=90.0°, γ=90.0°
      → 判定結果: 正方晶 (Tetragonal)
    
    Mg
      格子定数: a=3.21Å, b=3.21Å, c=5.21Å
      軸角: α=90.0°, β=90.0°, γ=120.0°
      → 判定結果: 六方晶 (Hexagonal)
    
    CaCO₃
      格子定数: a=4.99Å, b=4.99Å, c=4.99Å
      軸角: α=101.9°, β=101.9°, γ=101.9°
      → 判定結果: 三方晶 (Trigonal/Rhombohedral)
    
    石膏
      格子定数: a=5.68Å, b=15.18Å, c=6.29Å
      軸角: α=90.0°, β=113.8°, γ=90.0°
      → 判定結果: 単斜晶 (Monoclinic)
    
    硫黄
      格子定数: a=10.47Å, b=12.87Å, c=24.49Å
      軸角: α=90.0°, β=90.0°, γ=90.0°
      → 判定結果: 直方晶 (Orthorhombic)
    
    ================================================================================
    

**解説** : このプログラムにより、格子定数から自動的に結晶系を判定できます。実験データから得られた格子定数を入力すれば、結晶構造の分類が可能です。許容誤差（tolerance）を調整することで、測定誤差にも対応できます。

* * *

## 1.5 演習問題

学んだ内容を確認するために、以下の演習問題に取り組んでみましょう。

**演習1: 立方晶の単位格子体積計算**

**問題** : 鉄（α-Fe）の格子定数は a = 2.87 Åです。単位格子の体積を計算してください。

**解答例** :
    
    
    # 立方晶の単位格子体積計算
    a_Fe = 2.87  # Å
    V_Fe = a_Fe ** 3
    print(f"鉄（α-Fe）の単位格子体積: V = {V_Fe:.2f} ų")
    # 出力: 鉄（α-Fe）の単位格子体積: V = 23.64 ų
    

**演習2: 格子定数から結晶系判定**

**問題** : 以下の格子定数を持つ物質の結晶系を判定してください。

  * a = 4.5 Å, b = 4.5 Å, c = 6.0 Å
  * α = 90°, β = 90°, γ = 90°

**解答例** :
    
    
    # コード例8の関数を使用
    system = determine_crystal_system(4.5, 4.5, 6.0, 90, 90, 90)
    print(f"判定結果: {system}")
    # 出力: 判定結果: 正方晶 (Tetragonal)
    

**理由** : a = b ≠ c かつ α = β = γ = 90° なので、正方晶系です。

**演習3: 2D格子の格子点座標計算**

**問題** : 2次元正方格子（a = 3.0 Å）で、3×3の範囲にある格子点の座標をすべて列挙してください。

**解答例** :
    
    
    a = 3.0
    points = []
    for i in range(4):  # 0, 1, 2, 3
        for j in range(4):
            points.append((i * a, j * a))
    
    print("格子点の座標:")
    for i, (x, y) in enumerate(points, 1):
        print(f"  点{i:2d}: ({x:.1f}, {y:.1f})")
    print(f"\n総格子点数: {len(points)}")
    # 総格子点数: 16
    

**演習4: スーパーセルのサイズ計算**

**問題** : 立方晶（a = 5.0 Å）の単位格子を4×4×4倍に拡張したスーパーセルの体積と、単位格子内に1個の原子がある場合のスーパーセル内の原子総数を計算してください。

**解答例** :
    
    
    a = 5.0  # Å
    n = 4    # 拡張倍率
    
    # スーパーセルのサイズ
    supercell_size = n * a
    supercell_volume = supercell_size ** 3
    
    # 原子総数
    atoms_per_unit_cell = 1
    total_atoms = atoms_per_unit_cell * n ** 3
    
    print(f"スーパーセルのサイズ: {supercell_size} Å")
    print(f"スーパーセルの体積: {supercell_volume} ų")
    print(f"原子総数: {total_atoms} 個")
    # 出力:
    # スーパーセルのサイズ: 20.0 Å
    # スーパーセルの体積: 8000.0 ų
    # 原子総数: 64 個
    

**演習5: 結晶系の対称性比較**

**問題** : 立方晶、正方晶、直方晶の対称性を比較し、それぞれの回転対称軸の数を答えてください。

**解答** :

  * **立方晶** : 
    * 4回軸: 3本（x, y, z軸）
    * 3回軸: 4本（体対角線）
    * 2回軸: 6本（面対角線）
    * 最も高い対称性
  * **正方晶** : 
    * 4回軸: 1本（c軸）
    * 2回軸: 4本（a, b軸と対角線）
    * 中程度の対称性
  * **直方晶** : 
    * 2回軸: 3本（a, b, c軸）
    * 最も低い対称性（3つの中で）

**結論** : 立方晶 > 正方晶 > 直方晶 の順に対称性が高い。

* * *

## 1.6 本章のまとめ

### 学んだこと

  1. **結晶学の基本概念**
     * 結晶は原子が規則的に配列した物質（長距離秩序、周期性）
     * 非晶質は原子が不規則に配列した物質（短距離秩序のみ）
     * 周期性と対称性が結晶構造の基本
  2. **格子点と単位格子**
     * 格子点は周期的に繰り返される構造の基準点
     * 単位格子は結晶構造の最小繰り返し単位
     * 格子定数：a, b, c（軸長）と α, β, γ（軸角）の6つのパラメータ
  3. **7つの結晶系**
     * 三斜晶、単斜晶、直方晶、正方晶、六方晶、三方晶、立方晶
     * 格子定数の関係と対称性により分類される
     * 立方晶が最も対称性が高く、三斜晶が最も低い
  4. **Pythonによる可視化**
     * 格子点の生成と可視化
     * 単位格子の3D表示
     * スーパーセルの構築
     * 格子定数から結晶系を自動判定

### 重要なポイント

  * 結晶構造は**周期性と対称性** で特徴づけられる
  * 単位格子を繰り返すことで、無限に広がる結晶構造を構築できる
  * 格子定数は結晶構造を定量的に記述する基本パラメータ
  * 結晶では、**1, 2, 3, 4, 6回対称のみ** が許される（5回対称は不可）
  * 対称性が高いほど、格子定数間の制約が多い（自由度が低い）

### 次の章へ

第2章では、**ブラベー格子と空間群** を学びます：

  * 14種類のブラベー格子（プリミティブ、体心、面心、底心）
  * 空間群と対称操作（回転、反転、鏡映、らせん、映進）
  * 結晶構造の完全な記述方法
  * Pythonによるブラベー格子の可視化
  * 代表的な結晶構造（fcc, bcc, hcp, ダイヤモンド構造）
