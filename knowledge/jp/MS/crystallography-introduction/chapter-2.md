---
title: 第2章：ブラベー格子と空間群
chapter_title: 第2章：ブラベー格子と空間群
subtitle: 結晶の対称性を理解し、230種類の空間群の基礎を学ぶ
reading_time: 26-32分
difficulty: 初級〜中級
code_examples: 8
---

第1章で学んだ格子の概念を発展させ、この章では14種類のブラベー格子と230種類の空間群という結晶学の核心に迫ります。これらは、すべての結晶構造を分類する基本フレームワークであり、材料科学における最も重要な概念の一つです。Pythonライブラリ「pymatgen」を使った実践的な解析手法も習得します。 

## 学習目標

この章を読むことで、以下を習得できます：

  * ✅ 14種類のブラベー格子を分類・識別できる
  * ✅ 並進対称性と点対称性の違いを理解する
  * ✅ 対称操作（回転、鏡映、反転、回反）の基本を理解する
  * ✅ 空間群の基本概念と国際記号の読み方を習得する
  * ✅ pymatgenで空間群情報を取得・解析できる
  * ✅ 実在材料の空間群を調査し、対称操作を理解できる

* * *

## 2.1 ブラベー格子とは

### 格子の分類方法

第1章で学んだように、結晶は**格子（lattice）** という周期的な配列を持ちます。しかし、すべての格子が同じ形をしているわけではありません。1848年、フランスの物理学者**オーギュスト・ブラベー（Auguste Bravais）** は、3次元空間における格子を系統的に分類し、**14種類のブラベー格子** のみが存在することを証明しました。

ブラベー格子は、以下の2つの基準で分類されます：

  1. **結晶系（crystal system）** : 格子の対称性に基づく7種類の分類
  2. **格子点の配置（centering type）** : 単位格子内の格子点の位置

### 格子点の配置パターン

単位格子内の格子点の配置には、以下の4つのタイプがあります：

記号 | 名称（日本語） | 名称（英語） | 格子点の位置  
---|---|---|---  
**P** | 単純格子 | Primitive | 頂点のみ（8箇所）  
**I** | 体心格子 | Body-centered (Innenzentriert) | 頂点 + 体心（中心）  
**F** | 面心格子 | Face-centered | 頂点 + 各面の中心（6箇所）  
**C/A/B** | 底心格子 | Base-centered | 頂点 + 1組の対面の中心  
  
**重要な注意点** : これらの配置は、「対称性を保ったまま単位格子を縮小できない」という条件を満たす必要があります。例えば、単斜晶系には体心（I）や面心（F）は存在しません（それらは単純格子（P）に簡約できるため）。

### 14種類のブラベー格子の存在

7つの結晶系と4つの配置タイプを組み合わせると、理論的には28種類の格子が考えられますが、実際には**14種類のみ** が独立に存在します。これは、一部の組み合わせが他の格子に変換可能だからです。

> **数学的背景** : ブラベー格子の14種類という数は、3次元ユークリッド空間における点群の対称性と並進対称性の組み合わせから導かれます。この分類は、群論における**結晶学的点群（crystallographic point groups）** の理論に基づいています。 

* * *

## 2.2 14種類のブラベー格子の詳細

以下に、7つの結晶系ごとにブラベー格子を分類します。各格子の特徴と実在材料の例を示します。
    
    
    ```mermaid
    graph TB
        A[14種類のブラベー格子] --> B[三斜晶系Triclinic1種]
        A --> C[単斜晶系Monoclinic2種]
        A --> D[直方晶系Orthorhombic4種]
        A --> E[正方晶系Tetragonal2種]
        A --> F[六方晶系Hexagonal1種]
        A --> G[三方晶系Trigonal1種]
        A --> H[立方晶系Cubic3種]
    
        B --> B1[P]
        C --> C1[P]
        C --> C2[C]
        D --> D1[P]
        D --> D2[C]
        D --> D3[I]
        D --> D4[F]
        E --> E1[P]
        E --> E2[I]
        F --> F1[P]
        G --> G1[P または R]
        H --> H1[Psimple cubic]
        H --> H2[IBCC]
        H --> H3[FFCC]
    
        style B fill:#fce7f3
        style C fill:#fce7f3
        style D fill:#fce7f3
        style E fill:#fce7f3
        style F fill:#fce7f3
        style G fill:#fce7f3
        style H fill:#fce7f3
    ```

### 1\. 三斜晶系（Triclinic）- 1種類

**格子パラメータ** : $a \neq b \neq c$, $\alpha \neq \beta \neq \gamma \neq 90°$（すべて異なる）

  * **P（単純）** : 最も低い対称性を持つ格子
  * **実例** : CuSO₄·5H₂O（硫酸銅五水和物）、K₂Cr₂O₇（重クロム酸カリウム）

### 2\. 単斜晶系（Monoclinic）- 2種類

**格子パラメータ** : $a \neq b \neq c$, $\alpha = \gamma = 90° \neq \beta$

  * **P（単純）** : 頂点のみ
  * **C（底心）** : c軸に垂直な面（ab面）の中心に格子点を追加
  * **実例** : β-S（斜方硫黄）、石膏（CaSO₄·2H₂O）

### 3\. 直方晶系（Orthorhombic）- 4種類

**格子パラメータ** : $a \neq b \neq c$, $\alpha = \beta = \gamma = 90°$

  * **P（単純）** : 頂点のみ
  * **C（底心）** : ab面の中心に格子点
  * **I（体心）** : 体心に格子点
  * **F（面心）** : すべての面の中心に格子点
  * **実例** : α-S（斜方硫黄、P）、BaSO₄（重晶石、P）、U（ウラン、C）

### 4\. 正方晶系（Tetragonal）- 2種類

**格子パラメータ** : $a = b \neq c$, $\alpha = \beta = \gamma = 90°$

  * **P（単純）** : 頂点のみ
  * **I（体心）** : 体心に格子点
  * **実例** : TiO₂（ルチル、P）、In（インジウム、I）、Sn（白色スズ、I）

### 5\. 六方晶系（Hexagonal）- 1種類

**格子パラメータ** : $a = b \neq c$, $\alpha = \beta = 90°$, $\gamma = 120°$

  * **P（単純）** : 六角形の頂点に格子点
  * **実例** : グラファイト（C）、Mg（マグネシウム）、Zn（亜鉛）、氷

**注意** : 六方晶系の結晶は、しばしば**六方最密充填（HCP: Hexagonal Close-Packed）** 構造を取ります。HCPは、ブラベー格子そのものではなく、格子に原子を配置した**結晶構造** です。

### 6\. 三方晶系（Trigonal/Rhombohedral）- 1種類

**格子パラメータ** : $a = b = c$, $\alpha = \beta = \gamma \neq 90°$

  * **P（または R）** : 菱面体格子（rhombohedral lattice）
  * **実例** : 方解石（CaCO₃）、水晶（α-SiO₂）、Bi（ビスマス）

**六方晶系との関係** : 三方晶系の結晶は、六方晶系の設定でも記述できます。結晶学では、しばしば六方軸を用いて表記されます。

### 7\. 立方晶系（Cubic）- 3種類

**格子パラメータ** : $a = b = c$, $\alpha = \beta = \gamma = 90°$

  * **P（単純立方格子、simple cubic）** : 頂点のみ（実在材料は非常に少ない）
  * **I（体心立方格子、BCC: Body-Centered Cubic）** : 体心に格子点
  * **F（面心立方格子、FCC: Face-Centered Cubic）** : すべての面の中心に格子点

**実例** :

  * **P（simple cubic）** : α-Po（ポロニウム）、CsCl型構造（ただしCsCl自体は2原子基底）
  * **I（BCC）** : Fe（α鉄、常温）、Cr（クロム）、W（タングステン）、Na（ナトリウム）
  * **F（FCC）** : Al（アルミニウム）、Cu（銅）、Au（金）、Ag（銀）、Ni（ニッケル）、Pb（鉛）

### 14種類のブラベー格子まとめ表

結晶系 | 格子パラメータの条件 | ブラベー格子 | 合計  
---|---|---|---  
三斜晶系 | $a \neq b \neq c$, $\alpha \neq \beta \neq \gamma$ | P | 1  
単斜晶系 | $a \neq b \neq c$, $\alpha = \gamma = 90° \neq \beta$ | P, C | 2  
直方晶系 | $a \neq b \neq c$, $\alpha = \beta = \gamma = 90°$ | P, C, I, F | 4  
正方晶系 | $a = b \neq c$, $\alpha = \beta = \gamma = 90°$ | P, I | 2  
六方晶系 | $a = b \neq c$, $\alpha = \beta = 90°, \gamma = 120°$ | P | 1  
三方晶系 | $a = b = c$, $\alpha = \beta = \gamma \neq 90°$ | P (R) | 1  
立方晶系 | $a = b = c$, $\alpha = \beta = \gamma = 90°$ | P, I, F | 3  
**合計** | **14**  
  
* * *

## 2.3 対称操作

結晶の対称性は、**対称操作（symmetry operation）** によって記述されます。対称操作とは、結晶を特定の変換（回転、鏡映など）によって移動させても、元の配置と区別がつかない操作のことです。

対称操作は、大きく2つのカテゴリに分類されます：

  1. **点対称操作（point symmetry operations）** : 少なくとも1点が不動（移動しない）
  2. **並進対称操作（translational symmetry operations）** : すべての点が移動する

### 点対称操作

#### 1\. 回転（Rotation）

ある軸の周りに特定の角度だけ回転させる操作です。結晶学では、**1回、2回、3回、4回、6回回転対称軸** のみが許されます（5回や7回以上は、並進対称性と両立しません）。

回転対称 | 記号 | 回転角 | 繰り返し回数  
---|---|---|---  
1回回転 | 1 | 360° | 1（恒等操作）  
2回回転 | 2 | 180° | 2  
3回回転 | 3 | 120° | 3  
4回回転 | 4 | 90° | 4  
6回回転 | 6 | 60° | 6  
  
> **結晶学的制限（Crystallographic restriction）** : なぜ5回や7回回転対称は存在しないのでしょうか？これは、並進対称性（格子の周期性）と両立する回転対称が、1, 2, 3, 4, 6回のみであることが数学的に証明されているからです。5回対称は、準結晶（quasicrystal）にのみ現れます。 

#### 2\. 鏡映（Mirror reflection）

ある平面に関して鏡像対称を持つ操作です。記号は**m** で表されます。

#### 3\. 反転（Inversion）

ある点（反転中心）に関して、すべての座標を反転させる操作です。記号は**$\bar{1}$** または**i** で表されます。

反転操作の数式表現: $(x, y, z) \rightarrow (-x, -y, -z)$

#### 4\. 回反（Rotoinversion）

回転と反転を組み合わせた操作です。記号は**$\bar{1}, \bar{2}, \bar{3}, \bar{4}, \bar{6}$** で表されます。

例えば、$\bar{4}$は「90°回転した後、反転する」操作です。

### 並進対称操作

並進対称操作は、回転や鏡映に並進（平行移動）を組み合わせた操作です。

#### 1\. 映進（Glide reflection）

鏡映と並進を組み合わせた操作です。記号は**a, b, c, n, d** などで表されます。

例: **a-glide** は、ある平面に関する鏡映の後、a軸方向に $a/2$ だけ並進する操作です。

#### 2\. らせん（Screw axis）

回転と並進を組み合わせた操作です。記号は**2₁, 3₁, 4₁, 6₁** などで表されます。

例: **2₁** は、180°回転した後、回転軸方向に格子ベクトルの半分だけ並進する操作です（「21-screw」と読みます）。

### Hermann-Mauguin記号

対称操作の組み合わせは、**Hermann-Mauguin記号（国際記号）** で表記されます。これは、結晶の点群や空間群を記述する標準的な方法です。

例:

  * **2/m** : 2回回転軸と、それに垂直な鏡映面
  * **4mm** : 4回回転軸と、2つの鏡映面
  * **$\bar{3}$m** : 3回回反軸と鏡映面

* * *

## 2.4 空間群の概念

### 空間群とは

**空間群（space group）** は、結晶のすべての対称操作（点対称操作 + 並進対称操作）の集合です。1891年、ロシアの結晶学者**エフグラフ・フェドロフ（Evgraf Fedorov）** と、ドイツの数学者**アルトゥル・シェーンフリース（Arthur Schönflies）** が独立に、3次元空間には**230種類の空間群** のみが存在することを証明しました。

これは驚くべき結果です：すべての結晶構造（無限に多様）は、わずか230種類の対称性パターンのいずれかに分類できるのです。

### 点群と空間群の関係

空間群を理解するには、まず**点群（point group）** を理解する必要があります。

  * **点群** : 点対称操作のみの集合（並進は含まない）→ **32種類** が存在
  * **空間群** : 点対称操作 + 並進対称操作の集合 → **230種類** が存在

各空間群は、1つの点群に対応します。230種類の空間群は、32種類の点群に基づいて構築されています。

### 空間群の番号と国際記号

230種類の空間群には、1番から230番までの**番号** と、Hermann-Mauguin記号による**国際記号** が割り当てられています。

**代表的な空間群の例** :

番号 | 国際記号 | 結晶系 | 実例  
---|---|---|---  
1 | P1 | 三斜晶系 | 最も低い対称性  
2 | P$\bar{1}$ | 三斜晶系 | 反転中心を持つ  
63 | Cmcm | 直方晶系 | 底心 + 鏡映 + c-glide + 鏡映  
139 | I4/mmm | 正方晶系 | 体心 + 4回回転 + 鏡映  
194 | P6₃/mmc | 六方晶系 | Mg, Zn, Ti（HCP構造）  
225 | Fm$\bar{3}$m | 立方晶系 | Al, Cu, Au, Ag（FCC構造）  
229 | Im$\bar{3}$m | 立方晶系 | Fe, Cr, W（BCC構造）  
  
### 国際記号の読み方

空間群の国際記号は、以下のように構成されています：

**記号の構造** : `[格子タイプ][対称要素1][対称要素2][対称要素3]...`

例: **Fm$\bar{3}$m** （空間群225番）

  * **F** : 面心格子（Face-centered）
  * **m** : 鏡映面
  * **$\bar{3}$** : 3回回反軸
  * **m** : 別の鏡映面

この空間群は、銅（Cu）やアルミニウム（Al）などのFCC構造を持つ金属に対応します。

### 一般位置と特殊位置

空間群には、原子が配置できる位置として**一般位置（general position）** と**特殊位置（special position/Wyckoff position）** があります。

  * **一般位置** : 対称操作により、複数の等価な位置が生成される
  * **特殊位置** : 対称要素上にあるため、等価な位置の数が少ない

例えば、Fm$\bar{3}$m（FCC）の一般位置は48倍に増殖しますが、特殊位置（例: (0, 0, 0)）は1倍のままです。

* * *

## 2.5 Pythonによるブラベー格子と空間群の解析

ここからは、Pythonライブラリ**pymatgen（Python Materials Genomics）** を使って、ブラベー格子と空間群を実践的に解析します。

### pymatgenのインストール

pymatgenは、材料科学のための強力なPythonライブラリです。結晶構造の操作、対称性解析、Materials Project APIとの連携など、多くの機能を提供します。
    
    
    # pymatgenのインストール
    pip install pymatgen
    
    # 必要に応じて、以下もインストール
    pip install matplotlib numpy pandas plotly

**注意事項** :

  * pymatgenは依存関係が多いため、インストールに時間がかかる場合があります
  * 最新版（2024年以降）では、一部のAPIが変更されています
  * Materials Project APIを使う場合は、無料のAPIキーが必要です（後述）

### コード例1: 14種類のブラベー格子の情報テーブル

まず、14種類のブラベー格子の基本情報を整理したテーブルを作成します。
    
    
    import pandas as pd
    
    # 14種類のブラベー格子のデータ
    bravais_lattices = [
        # 三斜晶系
        {'結晶系': '三斜晶系', 'Bravais記号': 'aP', '格子タイプ': 'P (Primitive)',
         '格子点数/単位格子': 1, '対称性': '最低', '実例': 'CuSO₄·5H₂O'},
    
        # 単斜晶系
        {'結晶系': '単斜晶系', 'Bravais記号': 'mP', '格子タイプ': 'P (Primitive)',
         '格子点数/単位格子': 1, '対称性': '低', '実例': '石膏'},
        {'結晶系': '単斜晶系', 'Bravais記号': 'mC', '格子タイプ': 'C (Base-centered)',
         '格子点数/単位格子': 2, '対称性': '低', '実例': 'β-S'},
    
        # 直方晶系
        {'結晶系': '直方晶系', 'Bravais記号': 'oP', '格子タイプ': 'P (Primitive)',
         '格子点数/単位格子': 1, '対称性': '中', '実例': 'α-S'},
        {'結晶系': '直方晶系', 'Bravais記号': 'oC', '格子タイプ': 'C (Base-centered)',
         '格子点数/単位格子': 2, '対称性': '中', '実例': 'U'},
        {'結晶系': '直方晶系', 'Bravais記号': 'oI', '格子タイプ': 'I (Body-centered)',
         '格子点数/単位格子': 2, '対称性': '中', '実例': 'TiO₂ (brookite)'},
        {'結晶系': '直方晶系', 'Bravais記号': 'oF', '格子タイプ': 'F (Face-centered)',
         '格子点数/単位格子': 4, '対称性': '中', '実例': 'NaNO₃'},
    
        # 正方晶系
        {'結晶系': '正方晶系', 'Bravais記号': 'tP', '格子タイプ': 'P (Primitive)',
         '格子点数/単位格子': 1, '対称性': '高', '実例': 'TiO₂ (rutile)'},
        {'結晶系': '正方晶系', 'Bravais記号': 'tI', '格子タイプ': 'I (Body-centered)',
         '格子点数/単位格子': 2, '対称性': '高', '実例': 'In, Sn (白色)'},
    
        # 六方晶系
        {'結晶系': '六方晶系', 'Bravais記号': 'hP', '格子タイプ': 'P (Primitive)',
         '格子点数/単位格子': 1, '対称性': '高', '実例': 'Mg, Zn, グラファイト'},
    
        # 三方晶系
        {'結晶系': '三方晶系', 'Bravais記号': 'hR', '格子タイプ': 'R (Rhombohedral)',
         '格子点数/単位格子': 1, '対称性': '高', '実例': '方解石, Bi'},
    
        # 立方晶系
        {'結晶系': '立方晶系', 'Bravais記号': 'cP', '格子タイプ': 'P (Simple cubic)',
         '格子点数/単位格子': 1, '対称性': '最高', '実例': 'α-Po'},
        {'結晶系': '立方晶系', 'Bravais記号': 'cI', '格子タイプ': 'I (BCC)',
         '格子点数/単位格子': 2, '対称性': '最高', '実例': 'Fe, Cr, W'},
        {'結晶系': '立方晶系', 'Bravais記号': 'cF', '格子タイプ': 'F (FCC)',
         '格子点数/単位格子': 4, '対称性': '最高', '実例': 'Al, Cu, Au, Ag'},
    ]
    
    # DataFrameに変換
    df = pd.DataFrame(bravais_lattices)
    
    print("=" * 100)
    print("14種類のブラベー格子の分類")
    print("=" * 100)
    print(df.to_string(index=False))
    
    # 結晶系ごとの集計
    print("\n" + "=" * 100)
    print("結晶系ごとのブラベー格子の数")
    print("=" * 100)
    print(df.groupby('結晶系').size())
    
    print("\n【解説】")
    print("- Bravais記号: Pearson記号とも呼ばれ、結晶系の頭文字 + 格子タイプで表記")
    print("- 格子点数: 単位格子内の格子点の数（P=1, C/I=2, F=4）")
    print("- 対称性が高いほど、格子パラメータの制約が強くなる")
    print("- 立方晶系（cP, cI, cF）が最も対称性が高く、材料科学で重要")

**出力例** :
    
    
    ====================================================================================================
    14種類のブラベー格子の分類
    ====================================================================================================
      結晶系 Bravais記号           格子タイプ  格子点数/単位格子 対称性                実例
    三斜晶系         aP     P (Primitive)              1   最低       CuSO₄·5H₂O
    単斜晶系         mP     P (Primitive)              1     低               石膏
    単斜晶系         mC  C (Base-centered)              2     低               β-S
    直方晶系         oP     P (Primitive)              1     中               α-S
    直方晶系         oC  C (Base-centered)              2     中                 U
    直方晶系         oI  I (Body-centered)              2     中  TiO₂ (brookite)
    直方晶系         oF  F (Face-centered)              4     中            NaNO₃
    正方晶系         tP     P (Primitive)              1     高   TiO₂ (rutile)
    正方晶系         tI  I (Body-centered)              2     高      In, Sn (白色)
    六方晶系         hP     P (Primitive)              1     高   Mg, Zn, グラファイト
    三方晶系         hR  R (Rhombohedral)              1     高         方解石, Bi
    立方晶系         cP  P (Simple cubic)              1   最高             α-Po
    立方晶系         cI         I (BCC)              2   最高          Fe, Cr, W
    立方晶系         cF         F (FCC)              4   最高      Al, Cu, Au, Ag
    
    ====================================================================================================
    結晶系ごとのブラベー格子の数
    ====================================================================================================
    結晶系
    三斜晶系    1
    三方晶系    1
    六方晶系    1
    単斜晶系    2
    正方晶系    2
    直方晶系    4
    立方晶系    3
    dtype: int64

### コード例2: 立方晶3種のブラベー格子可視化

最も重要な立方晶系の3つのブラベー格子（P, I, F）を3Dで可視化します。
    
    
    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    
    def create_cubic_lattice_points(lattice_type='P', a=1.0):
        """
        立方格子の格子点座標を生成する関数
    
        Parameters:
        -----------
        lattice_type : str
            'P' (Primitive), 'I' (Body-centered), 'F' (Face-centered)
        a : float
            格子定数
    
        Returns:
        --------
        np.ndarray : 格子点の座標 (N, 3)
        """
        # 頂点（8箇所）
        corners = np.array([
            [0, 0, 0], [a, 0, 0], [a, a, 0], [0, a, 0],
            [0, 0, a], [a, 0, a], [a, a, a], [0, a, a]
        ])
    
        if lattice_type == 'P':
            return corners
    
        elif lattice_type == 'I':
            # 体心を追加
            body_center = np.array([[a/2, a/2, a/2]])
            return np.vstack([corners, body_center])
    
        elif lattice_type == 'F':
            # 6つの面の中心を追加
            face_centers = np.array([
                [a/2, a/2, 0],   # bottom face
                [a/2, a/2, a],   # top face
                [a/2, 0, a/2],   # front face
                [a/2, a, a/2],   # back face
                [0, a/2, a/2],   # left face
                [a, a/2, a/2]    # right face
            ])
            return np.vstack([corners, face_centers])
    
        else:
            raise ValueError("lattice_type must be 'P', 'I', or 'F'")
    
    
    def plot_cubic_unit_cell(ax, lattice_type='P', a=1.0):
        """
        立方格子の単位格子をプロットする関数
        """
        # 格子点の生成
        points = create_cubic_lattice_points(lattice_type, a)
    
        # 格子点をプロット
        ax.add_trace(go.Scatter3d(
            x=points[:, 0], y=points[:, 1], z=points[:, 2],
            mode='markers',
            marker=dict(size=8, color='red', symbol='circle'),
            name='格子点'
        ))
    
        # 単位格子の辺を描画
        edges = [
            [0, 1], [1, 2], [2, 3], [3, 0],  # bottom
            [4, 5], [5, 6], [6, 7], [7, 4],  # top
            [0, 4], [1, 5], [2, 6], [3, 7]   # vertical
        ]
    
        corners = create_cubic_lattice_points('P', a)
    
        for edge in edges:
            ax.add_trace(go.Scatter3d(
                x=corners[edge, 0], y=corners[edge, 1], z=corners[edge, 2],
                mode='lines',
                line=dict(color='black', width=2),
                showlegend=False
            ))
    
    
    # 3つのサブプロット作成
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=('(a) Simple Cubic (P)', '(b) Body-Centered Cubic (I)', '(c) Face-Centered Cubic (F)'),
        specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}, {'type': 'scatter3d'}]],
        horizontal_spacing=0.05
    )
    
    # 各格子タイプをプロット
    for i, lattice_type in enumerate(['P', 'I', 'F'], start=1):
        points = create_cubic_lattice_points(lattice_type, a=1.0)
    
        # 格子点
        fig.add_trace(go.Scatter3d(
            x=points[:, 0], y=points[:, 1], z=points[:, 2],
            mode='markers',
            marker=dict(size=6, color='red'),
            showlegend=False
        ), row=1, col=i)
    
        # 単位格子の辺
        edges = [
            [0, 1], [1, 2], [2, 3], [3, 0],
            [4, 5], [5, 6], [6, 7], [7, 4],
            [0, 4], [1, 5], [2, 6], [3, 7]
        ]
        corners = create_cubic_lattice_points('P', a=1.0)
    
        for edge in edges:
            fig.add_trace(go.Scatter3d(
                x=corners[edge, 0], y=corners[edge, 1], z=corners[edge, 2],
                mode='lines',
                line=dict(color='black', width=2),
                showlegend=False
            ), row=1, col=i)
    
    # レイアウト設定
    fig.update_layout(
        title_text="立方晶系の3つのブラベー格子",
        height=500,
        showlegend=False
    )
    
    # 各軸の設定
    for i in range(1, 4):
        fig.update_scenes(
            xaxis=dict(range=[0, 1], title='x'),
            yaxis=dict(range=[0, 1], title='y'),
            zaxis=dict(range=[0, 1], title='z'),
            aspectmode='cube',
            row=1, col=i
        )
    
    fig.show()
    
    # 格子点数の比較
    print("\n【立方晶系の3つのブラベー格子】")
    print("=" * 60)
    for lattice_type, name, example in [('P', 'Simple Cubic', 'α-Po'),
                                          ('I', 'BCC', 'Fe, Cr, W'),
                                          ('F', 'FCC', 'Al, Cu, Au, Ag')]:
        points = create_cubic_lattice_points(lattice_type)
        print(f"{name:20s} | 格子点数: {len(points):2d} | 実例: {example}")
    print("=" * 60)
    print("注: 頂点の格子点は8つの単位格子で共有されるため、実効的には1/8ずつカウント")
    print("    P: 8×(1/8) = 1, I: 8×(1/8) + 1 = 2, F: 8×(1/8) + 6×(1/2) = 4")

### コード例3: 直方晶4種のブラベー格子比較

直方晶系の4つのブラベー格子（P, C, I, F）の違いを可視化します。
    
    
    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    
    def create_orthorhombic_lattice_points(lattice_type='P', a=1.0, b=1.2, c=0.8):
        """
        直方晶格子の格子点座標を生成する関数
    
        Parameters:
        -----------
        lattice_type : str
            'P', 'C', 'I', 'F'
        a, b, c : float
            格子定数（a ≠ b ≠ c）
        """
        # 頂点（8箇所）
        corners = np.array([
            [0, 0, 0], [a, 0, 0], [a, b, 0], [0, b, 0],
            [0, 0, c], [a, 0, c], [a, b, c], [0, b, c]
        ])
    
        if lattice_type == 'P':
            return corners
    
        elif lattice_type == 'C':
            # C-centered: ab面の中心（z=0とz=cの2箇所）
            base_centers = np.array([
                [a/2, b/2, 0],
                [a/2, b/2, c]
            ])
            return np.vstack([corners, base_centers])
    
        elif lattice_type == 'I':
            # Body-centered: 体心
            body_center = np.array([[a/2, b/2, c/2]])
            return np.vstack([corners, body_center])
    
        elif lattice_type == 'F':
            # Face-centered: 6つの面の中心
            face_centers = np.array([
                [a/2, b/2, 0],   # z=0 face
                [a/2, b/2, c],   # z=c face
                [a/2, 0, c/2],   # y=0 face
                [a/2, b, c/2],   # y=b face
                [0, b/2, c/2],   # x=0 face
                [a, b/2, c/2]    # x=a face
            ])
            return np.vstack([corners, face_centers])
    
        else:
            raise ValueError("lattice_type must be 'P', 'C', 'I', or 'F'")
    
    
    # 4つのサブプロット作成
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('(a) Primitive (P)', '(b) C-centered (C)',
                        '(c) Body-centered (I)', '(d) Face-centered (F)'),
        specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}],
               [{'type': 'scatter3d'}, {'type': 'scatter3d'}]],
        horizontal_spacing=0.05,
        vertical_spacing=0.1
    )
    
    a, b, c = 1.0, 1.3, 0.7
    
    positions = [(1, 1), (1, 2), (2, 1), (2, 2)]
    lattice_types = ['P', 'C', 'I', 'F']
    
    for (row, col), lattice_type in zip(positions, lattice_types):
        points = create_orthorhombic_lattice_points(lattice_type, a, b, c)
    
        # 格子点
        fig.add_trace(go.Scatter3d(
            x=points[:, 0], y=points[:, 1], z=points[:, 2],
            mode='markers',
            marker=dict(size=5, color='red'),
            showlegend=False
        ), row=row, col=col)
    
        # 単位格子の辺
        edges = [
            [0, 1], [1, 2], [2, 3], [3, 0],
            [4, 5], [5, 6], [6, 7], [7, 4],
            [0, 4], [1, 5], [2, 6], [3, 7]
        ]
        corners = create_orthorhombic_lattice_points('P', a, b, c)
    
        for edge in edges:
            fig.add_trace(go.Scatter3d(
                x=corners[edge, 0], y=corners[edge, 1], z=corners[edge, 2],
                mode='lines',
                line=dict(color='black', width=2),
                showlegend=False
            ), row=row, col=col)
    
    # レイアウト設定
    fig.update_layout(
        title_text="直方晶系の4つのブラベー格子（a ≠ b ≠ c, α = β = γ = 90°）",
        height=800,
        showlegend=False
    )
    
    fig.show()
    
    # 格子点数の比較
    print("\n【直方晶系の4つのブラベー格子】")
    print("=" * 70)
    print(f"{'格子タイプ':15s} | {'格子点数':8s} | {'実効格子点数':12s} | 実例")
    print("=" * 70)
    for lt, name, eff, example in [
        ('P', 'Primitive', 1, 'α-S'),
        ('C', 'C-centered', 2, 'U'),
        ('I', 'Body-centered', 2, 'TiO₂ (brookite)'),
        ('F', 'Face-centered', 4, 'NaNO₃')
    ]:
        points = create_orthorhombic_lattice_points(lt, a, b, c)
        print(f"{name:15s} | {len(points):8d} | {eff:12d} | {example}")
    print("=" * 70)

### コード例4: pymatgenで空間群情報取得

ここから、pymatgenを使った実践的な空間群解析を行います。まず、pymatgenで空間群の基本情報を取得する方法を学びます。
    
    
    from pymatgen.symmetry.groups import SpaceGroup
    
    # 代表的な空間群の情報を取得
    space_groups = [
        1,    # P1 (三斜晶系、最低対称性)
        2,    # P-1 (反転中心あり)
        194,  # P6₃/mmc (HCP構造)
        225,  # Fm-3m (FCC構造)
        229   # Im-3m (BCC構造)
    ]
    
    print("=" * 100)
    print("代表的な空間群の詳細情報")
    print("=" * 100)
    
    for sg_num in space_groups:
        sg = SpaceGroup.from_int_number(sg_num)
    
        print(f"\n【空間群 {sg_num}番】")
        print(f"  国際記号:     {sg.symbol}")
        print(f"  結晶系:       {sg.crystal_system}")
        print(f"  点群:         {sg.point_group}")
        print(f"  対称操作数:   {len(sg.symmetry_ops)}")
        print(f"  centrosymmetric: {sg.is_centrosymmetric} (反転中心の有無)")
    
        # 対称操作の一部を表示（最初の5個）
        print(f"  対称操作（最初の5個）:")
        for i, op in enumerate(sg.symmetry_ops[:5], 1):
            print(f"    {i}. {op}")
    
        if len(sg.symmetry_ops) > 5:
            print(f"    ... 他 {len(sg.symmetry_ops) - 5} 個")
    
    print("\n" + "=" * 100)
    print("【解説】")
    print("- 対称操作数は、空間群によって異なります（1番: 1個、225番: 192個）")
    print("- centrosymmetric（中心対称）は、反転中心の有無を示します")
    print("- 対称操作は、回転行列と並進ベクトルで表現されます")
    print("=" * 100)

**出力例** :
    
    
    ====================================================================================================
    代表的な空間群の詳細情報
    ====================================================================================================
    
    【空間群 1番】
      国際記号:     P1
      結晶系:       triclinic
      点群:         1
      対称操作数:   1
      centrosymmetric: False (反転中心の有無)
      対称操作（最初の5個）:
        1. Rot:
    [[1. 0. 0.]
     [0. 1. 0.]
     [0. 0. 1.]]
    tau
    [0. 0. 0.]
    
    【空間群 225番】
      国際記号:     Fm-3m
      結晶系:       cubic
      点群:         m-3m
      対称操作数:   192
      centrosymmetric: True (反転中心の有無)
      対称操作（最初の5個）:
        1. Rot:
    [[1. 0. 0.]
     [0. 1. 0.]
     [0. 0. 1.]]
    tau
    [0. 0. 0.]
        2. Rot:
    [[-1.  0.  0.]
     [ 0. -1.  0.]
     [ 0.  0.  1.]]
    tau
    [0. 0. 0.]
        ... 他 187 個

### コード例5: 特定空間群の対称操作リスト

空間群225（Fm-3m, FCC構造）の対称操作を詳しく調べます。
    
    
    from pymatgen.symmetry.groups import SpaceGroup
    import numpy as np
    
    # 空間群225番（Fm-3m, FCC構造）
    sg = SpaceGroup.from_int_number(225)
    
    print("=" * 100)
    print(f"空間群 {sg.int_number}: {sg.symbol} の対称操作リスト")
    print("=" * 100)
    
    print(f"結晶系: {sg.crystal_system}")
    print(f"点群: {sg.point_group}")
    print(f"対称操作の総数: {len(sg.symmetry_ops)}")
    print(f"反転中心: {'あり' if sg.is_centrosymmetric else 'なし'}")
    
    print("\n【対称操作の分類】")
    
    # 対称操作の分類
    pure_translations = []
    rotations = []
    reflections = []
    inversions = []
    other_ops = []
    
    for i, op in enumerate(sg.symmetry_ops, 1):
        rotation_matrix = op.rotation_matrix
        translation_vector = op.translation_vector
    
        # 恒等操作
        if np.allclose(rotation_matrix, np.eye(3)) and np.allclose(translation_vector, 0):
            continue  # スキップ（恒等操作）
    
        # 純粋な並進
        if np.allclose(rotation_matrix, np.eye(3)):
            pure_translations.append((i, op))
    
        # 反転（-I）
        elif np.allclose(rotation_matrix, -np.eye(3)):
            inversions.append((i, op))
    
        # その他の操作
        else:
            # 行列式から回転/鏡映を判定
            det = np.linalg.det(rotation_matrix)
            if np.isclose(det, 1):
                rotations.append((i, op))
            elif np.isclose(det, -1):
                reflections.append((i, op))
            else:
                other_ops.append((i, op))
    
    print(f"  恒等操作:        1個")
    print(f"  純粋な並進:      {len(pure_translations)}個")
    print(f"  回転操作:        {len(rotations)}個")
    print(f"  鏡映操作:        {len(reflections)}個")
    print(f"  反転操作:        {len(inversions)}個")
    print(f"  その他:          {len(other_ops)}個")
    print(f"  合計:            {1 + len(pure_translations) + len(rotations) + len(reflections) + len(inversions) + len(other_ops)}個")
    
    # 純粋な並進の詳細表示
    print("\n【純粋な並進操作（最初の10個）】")
    for i, (num, op) in enumerate(pure_translations[:10], 1):
        print(f"  {num:3d}. 並進ベクトル: {op.translation_vector}")
    
    # 回転操作の詳細表示（最初の5個）
    print("\n【回転操作（最初の5個）】")
    for i, (num, op) in enumerate(rotations[:5], 1):
        print(f"  {num:3d}. 回転行列:")
        print(f"       {op.rotation_matrix}")
        print(f"       並進: {op.translation_vector}")
    
    print("\n" + "=" * 100)
    print("【解説】")
    print("- Fm-3m（FCC）は、立方晶系の中で最も高い対称性を持つ")
    print("- 192個の対称操作は、48個の点対称操作 × 4個の並進（FCC格子の特性）")
    print("- この高い対称性により、物性（弾性率、光学特性など）が等方的になる")
    print("=" * 100)

### コード例6: 結晶構造からの空間群判定

既知の結晶構造から、pymatgenが自動的に空間群を判定する例を示します。
    
    
    from pymatgen.core import Structure, Lattice
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
    
    # 例1: FCC構造のアルミニウム（Al）
    # 格子定数: a = 4.05 Å
    lattice_al = Lattice.cubic(4.05)
    
    # FCC構造: 原子は (0,0,0), (0.5,0.5,0), (0.5,0,0.5), (0,0.5,0.5) に配置
    al_structure = Structure(
        lattice_al,
        ["Al", "Al", "Al", "Al"],
        [[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]]
    )
    
    # 空間群解析
    sga_al = SpacegroupAnalyzer(al_structure)
    
    print("=" * 100)
    print("【例1: アルミニウム（Al）の結晶構造解析】")
    print("=" * 100)
    print(f"格子定数: a = 4.05 Å (立方晶)")
    print(f"原子数: {len(al_structure)}")
    print(f"\n空間群番号:       {sga_al.get_space_group_number()}")
    print(f"空間群記号:       {sga_al.get_space_group_symbol()}")
    print(f"点群:             {sga_al.get_point_group_symbol()}")
    print(f"結晶系:           {sga_al.get_crystal_system()}")
    print(f"格子タイプ:       {sga_al.get_lattice_type()}")
    
    # 例2: BCC構造の鉄（Fe）
    lattice_fe = Lattice.cubic(2.87)
    
    # BCC構造: 原子は (0,0,0) と (0.5,0.5,0.5) に配置
    fe_structure = Structure(
        lattice_fe,
        ["Fe", "Fe"],
        [[0, 0, 0], [0.5, 0.5, 0.5]]
    )
    
    sga_fe = SpacegroupAnalyzer(fe_structure)
    
    print("\n" + "=" * 100)
    print("【例2: 鉄（Fe）の結晶構造解析】")
    print("=" * 100)
    print(f"格子定数: a = 2.87 Å (立方晶)")
    print(f"原子数: {len(fe_structure)}")
    print(f"\n空間群番号:       {sga_fe.get_space_group_number()}")
    print(f"空間群記号:       {sga_fe.get_space_group_symbol()}")
    print(f"点群:             {sga_fe.get_point_group_symbol()}")
    print(f"結晶系:           {sga_fe.get_crystal_system()}")
    print(f"格子タイプ:       {sga_fe.get_lattice_type()}")
    
    # 例3: HCP構造のマグネシウム（Mg）
    # HCP: a = 3.21 Å, c = 5.21 Å, c/a = 1.624
    lattice_mg = Lattice.hexagonal(a=3.21, c=5.21)
    
    # HCP構造: 原子は (0,0,0), (1/3, 2/3, 1/2) に配置
    mg_structure = Structure(
        lattice_mg,
        ["Mg", "Mg"],
        [[0, 0, 0], [1/3, 2/3, 0.5]]
    )
    
    sga_mg = SpacegroupAnalyzer(mg_structure)
    
    print("\n" + "=" * 100)
    print("【例3: マグネシウム（Mg）の結晶構造解析】")
    print("=" * 100)
    print(f"格子定数: a = 3.21 Å, c = 5.21 Å (六方晶)")
    print(f"c/a比: {5.21/3.21:.3f}")
    print(f"原子数: {len(mg_structure)}")
    print(f"\n空間群番号:       {sga_mg.get_space_group_number()}")
    print(f"空間群記号:       {sga_mg.get_space_group_symbol()}")
    print(f"点群:             {sga_mg.get_point_group_symbol()}")
    print(f"結晶系:           {sga_mg.get_crystal_system()}")
    print(f"格子タイプ:       {sga_mg.get_lattice_type()}")
    
    print("\n" + "=" * 100)
    print("【解説】")
    print("- Al（FCC）: 空間群225番 Fm-3m")
    print("- Fe（BCC）: 空間群229番 Im-3m")
    print("- Mg（HCP）: 空間群194番 P6₃/mmc")
    print("- pymatgenは、原子座標から自動的に空間群を判定できる")
    print("=" * 100)

### コード例7: 等価位置の計算

空間群の対称操作により、1つの原子位置から生成される等価位置を計算します。
    
    
    from pymatgen.symmetry.groups import SpaceGroup
    import numpy as np
    
    # 空間群225番（Fm-3m, FCC構造）
    sg = SpaceGroup.from_int_number(225)
    
    print("=" * 100)
    print(f"空間群 {sg.int_number}: {sg.symbol} における等価位置の生成")
    print("=" * 100)
    
    # 一般位置の例: (x, y, z) = (0.3, 0.2, 0.1)
    original_position = np.array([0.3, 0.2, 0.1])
    
    print(f"\n元の原子位置: ({original_position[0]:.2f}, {original_position[1]:.2f}, {original_position[2]:.2f})")
    print(f"\n対称操作により生成される等価位置:")
    
    # 等価位置を格納するセット（重複除去のため）
    equivalent_positions = set()
    
    for i, op in enumerate(sg.symmetry_ops, 1):
        # 対称操作を適用
        new_position = op.operate(original_position)
    
        # 周期境界条件を適用（0 ≤ coord < 1）
        new_position = new_position % 1.0
    
        # タプルに変換してセットに追加（重複チェックのため）
        pos_tuple = tuple(np.round(new_position, 6))
        equivalent_positions.add(pos_tuple)
    
    # 結果を表示
    print(f"\n等価位置の総数: {len(equivalent_positions)}個\n")
    
    for i, pos in enumerate(sorted(equivalent_positions), 1):
        print(f"  {i:3d}. ({pos[0]:7.4f}, {pos[1]:7.4f}, {pos[2]:7.4f})")
    
    # 特殊位置の例: (0, 0, 0) - 高対称点
    print("\n" + "=" * 100)
    print("【特殊位置の例: (0, 0, 0)】")
    print("=" * 100)
    
    special_position = np.array([0.0, 0.0, 0.0])
    special_positions = set()
    
    for op in sg.symmetry_ops:
        new_position = op.operate(special_position)
        new_position = new_position % 1.0
        pos_tuple = tuple(np.round(new_position, 6))
        special_positions.add(pos_tuple)
    
    print(f"等価位置の総数: {len(special_positions)}個")
    print("→ 特殊位置（高対称点）は、一般位置よりも少ない等価位置を生成します")
    
    print("\n" + "=" * 100)
    print("【解説】")
    print("- 一般位置（x, y, z）: 空間群の対称操作により、多数の等価位置が生成される")
    print("- 特殊位置（対称要素上）: 等価位置の数が少ない（例: (0,0,0)は1個のみ）")
    print("- Fm-3mの一般位置は192倍に増殖する（対称操作192個）")
    print("- 結晶構造の記述では、特殊位置を優先的に使うと原子数が少なくて済む")
    print("=" * 100)

### コード例8: 空間群番号から結晶例取得（概念）

最後に、Materials Project APIと連携して、特定の空間群を持つ実在材料を検索する方法を紹介します（APIキーが必要）。
    
    
    """
    Materials Project APIを使った空間群検索の例
    
    注意: このコードを実行するには、Materials Project APIキーが必要です。
    APIキーは、https://next-gen.materialsproject.org/ で無料登録後に取得できます。
    
    インストール:
    pip install mp-api
    """
    
    # 以下は概念的なコード例です（APIキー必要）
    
    from mp_api.client import MPRester
    
    def search_materials_by_space_group(space_group_number, api_key=None, max_results=5):
        """
        特定の空間群を持つ材料を検索する関数
    
        Parameters:
        -----------
        space_group_number : int
            空間群番号（1-230）
        api_key : str
            Materials Project APIキー
        max_results : int
            取得する最大結果数
    
        Returns:
        --------
        list : 検索結果（材料ID、化学式、空間群記号のリスト）
        """
        if api_key is None:
            print("エラー: APIキーが必要です")
            print("取得方法: https://next-gen.materialsproject.org/ で無料登録")
            return []
    
        with MPRester(api_key) as mpr:
            # 空間群で検索
            docs = mpr.materials.summary.search(
                spacegroup_number=space_group_number,
                fields=["material_id", "formula_pretty", "symmetry"],
                num_chunks=1
            )
    
            results = []
            for i, doc in enumerate(docs[:max_results]):
                results.append({
                    'material_id': doc.material_id,
                    'formula': doc.formula_pretty,
                    'space_group': doc.symmetry.symbol,
                    'crystal_system': doc.symmetry.crystal_system
                })
    
            return results
    
    
    # 使用例（APIキーを持っている場合）
    # API_KEY = "your_api_key_here"
    # results = search_materials_by_space_group(225, api_key=API_KEY, max_results=10)
    
    # APIキーがない場合のデモ出力
    print("=" * 100)
    print("【空間群225番（Fm-3m, FCC）を持つ代表的な材料】")
    print("=" * 100)
    
    # 既知の材料例（手動リスト）
    fcc_materials = [
        {'formula': 'Al', 'space_group': 'Fm-3m', 'crystal_system': 'cubic', 'description': 'アルミニウム'},
        {'formula': 'Cu', 'space_group': 'Fm-3m', 'crystal_system': 'cubic', 'description': '銅'},
        {'formula': 'Au', 'space_group': 'Fm-3m', 'crystal_system': 'cubic', 'description': '金'},
        {'formula': 'Ag', 'space_group': 'Fm-3m', 'crystal_system': 'cubic', 'description': '銀'},
        {'formula': 'Ni', 'space_group': 'Fm-3m', 'crystal_system': 'cubic', 'description': 'ニッケル'},
        {'formula': 'Pt', 'space_group': 'Fm-3m', 'crystal_system': 'cubic', 'description': '白金'},
        {'formula': 'Pb', 'space_group': 'Fm-3m', 'crystal_system': 'cubic', 'description': '鉛'},
        {'formula': 'NaCl', 'space_group': 'Fm-3m', 'crystal_system': 'cubic', 'description': '塩化ナトリウム'},
        {'formula': 'MgO', 'space_group': 'Fm-3m', 'crystal_system': 'cubic', 'description': '酸化マグネシウム'},
    ]
    
    for i, mat in enumerate(fcc_materials, 1):
        print(f"{i:2d}. {mat['formula']:10s} | {mat['space_group']:10s} | {mat['description']}")
    
    print("\n" + "=" * 100)
    print("【空間群229番（Im-3m, BCC）を持つ代表的な材料】")
    print("=" * 100)
    
    bcc_materials = [
        {'formula': 'Fe', 'space_group': 'Im-3m', 'crystal_system': 'cubic', 'description': '鉄（α相、常温）'},
        {'formula': 'Cr', 'space_group': 'Im-3m', 'crystal_system': 'cubic', 'description': 'クロム'},
        {'formula': 'W', 'space_group': 'Im-3m', 'crystal_system': 'cubic', 'description': 'タングステン'},
        {'formula': 'Mo', 'space_group': 'Im-3m', 'crystal_system': 'cubic', 'description': 'モリブデン'},
        {'formula': 'V', 'space_group': 'Im-3m', 'crystal_system': 'cubic', 'description': 'バナジウム'},
        {'formula': 'Na', 'space_group': 'Im-3m', 'crystal_system': 'cubic', 'description': 'ナトリウム'},
        {'formula': 'K', 'space_group': 'Im-3m', 'crystal_system': 'cubic', 'description': 'カリウム'},
    ]
    
    for i, mat in enumerate(bcc_materials, 1):
        print(f"{i:2d}. {mat['formula']:10s} | {mat['space_group']:10s} | {mat['description']}")
    
    print("\n" + "=" * 100)
    print("【解説】")
    print("- Materials Project API (https://next-gen.materialsproject.org/) を使うと、")
    print("  空間群番号から実在材料を検索できます（無料APIキー登録が必要）")
    print("- 上記の例は、手動でリストアップした代表的な材料です")
    print("- FCC（225番）とBCC（229番）は、金属材料で最も一般的な構造")
    print("=" * 100)

* * *

## 2.6 演習問題

**演習1: 14種類のブラベー格子の分類**

**問題** : 以下の格子パラメータを持つ結晶は、どの結晶系とどのブラベー格子に分類されますか？

  1. $a = b = c = 5.0$ Å, $\alpha = \beta = \gamma = 90°$, 格子点は頂点のみ
  2. $a = b = 3.2$ Å, $c = 5.1$ Å, $\alpha = \beta = 90°$, $\gamma = 120°$
  3. $a = 4.0$ Å, $b = 5.0$ Å, $c = 6.0$ Å, $\alpha = \beta = \gamma = 90°$, 体心に格子点
  4. $a = b = c = 4.05$ Å, $\alpha = \beta = \gamma = 90°$, すべての面の中心に格子点

**解答** :

  1. **立方晶系、P（simple cubic）** \- すべて等しく、90°、頂点のみ
  2. **六方晶系、P** \- $a = b \neq c$, $\gamma = 120°$
  3. **直方晶系、I（体心）** \- $a \neq b \neq c$, すべて90°、体心
  4. **立方晶系、F（FCC）** \- すべて等しく、90°、面心

**演習2: 立方晶の3つのブラベー格子の充填率比較**

**問題** : 立方晶系の3つのブラベー格子（P, I, F）について、原子を剛体球として扱ったときの充填率（Packing Fraction）を計算してください。格子定数を$a$、原子半径を$r$とします。

**ヒント** :

  * 充填率 = （原子の体積） / （単位格子の体積）
  * 原子の体積 = $\frac{4}{3}\pi r^3$
  * 単位格子の体積 = $a^3$
  * P: 原子は頂点のみ、最近接距離 = $a$
  * I（BCC）: 原子は頂点+体心、最近接距離 = $\frac{\sqrt{3}}{2}a$
  * F（FCC）: 原子は頂点+面心、最近接距離 = $\frac{a}{\sqrt{2}}$

**解答** :

**1\. Simple Cubic (P)** :

  * 単位格子あたりの原子数: $8 \times \frac{1}{8} = 1$
  * 最近接距離: $2r = a$ → $r = \frac{a}{2}$
  * 充填率: $\frac{1 \times \frac{4}{3}\pi r^3}{a^3} = \frac{\frac{4}{3}\pi (\frac{a}{2})^3}{a^3} = \frac{\pi}{6} \approx 0.524$ (52.4%)

**2\. BCC (I)** :

  * 単位格子あたりの原子数: $8 \times \frac{1}{8} + 1 = 2$
  * 最近接距離: $2r = \frac{\sqrt{3}}{2}a$ → $r = \frac{\sqrt{3}}{4}a$
  * 充填率: $\frac{2 \times \frac{4}{3}\pi r^3}{a^3} = \frac{2 \times \frac{4}{3}\pi (\frac{\sqrt{3}}{4}a)^3}{a^3} = \frac{\sqrt{3}\pi}{8} \approx 0.680$ (68.0%)

**3\. FCC (F)** :

  * 単位格子あたりの原子数: $8 \times \frac{1}{8} + 6 \times \frac{1}{2} = 4$
  * 最近接距離: $2r = \frac{a}{\sqrt{2}}$ → $r = \frac{a}{2\sqrt{2}}$
  * 充填率: $\frac{4 \times \frac{4}{3}\pi r^3}{a^3} = \frac{4 \times \frac{4}{3}\pi (\frac{a}{2\sqrt{2}})^3}{a^3} = \frac{\pi}{3\sqrt{2}} \approx 0.740$ (74.0%)

**結論** : FCC (74.0%) > BCC (68.0%) > Simple Cubic (52.4%)

**演習3: 対称操作の種類判定**

**問題** : 以下の対称操作は、どのタイプに分類されますか？

  1. z軸の周りに90°回転
  2. xy平面に関する鏡映
  3. 原点に関する反転: $(x, y, z) \rightarrow (-x, -y, -z)$
  4. 180°回転した後、回転軸方向に格子ベクトルの半分だけ並進
  5. 鏡映の後、鏡映面に平行に格子ベクトルの半分だけ並進

**解答** :

  1. **4回回転（4-fold rotation）** \- 記号: 4
  2. **鏡映（mirror reflection）** \- 記号: m
  3. **反転（inversion）** \- 記号: $\bar{1}$ または i
  4. **らせん（screw axis）** \- 記号: 2₁（2-fold screw）
  5. **映進（glide reflection）** \- 記号: a, b, c, n, d（方向により異なる）

**演習4: 空間群番号からHermann-Mauguin記号の読み方**

**問題** : 以下の空間群記号の意味を説明してください。

  1. P6₃/mmc (空間群194番)
  2. Fm$\bar{3}$m (空間群225番)
  3. Im$\bar{3}$m (空間群229番)

**解答** :

**1\. P6₃/mmc** :

  * **P** : 単純格子（Primitive）
  * **6₃** : 6回らせん軸（6-fold screw axis）
  * **/** : 垂直な対称要素
  * **m** : 鏡映面（mirror plane）
  * **m** : 別の鏡映面
  * **c** : c-glide（c軸方向の映進）
  * 六方晶系、HCP構造（Mg, Zn, Tiなど）に対応

**2\. Fm$\bar{3}$m** :

  * **F** : 面心格子（Face-centered）
  * **m** : 鏡映面
  * **$\bar{3}$** : 3回回反軸（3-fold rotoinversion）
  * **m** : 別の鏡映面
  * 立方晶系、FCC構造（Al, Cu, Au, Agなど）に対応

**3\. Im$\bar{3}$m** :

  * **I** : 体心格子（Body-centered）
  * **m** : 鏡映面
  * **$\bar{3}$** : 3回回反軸
  * **m** : 別の鏡映面
  * 立方晶系、BCC構造（Fe, Cr, Wなど）に対応

**演習5: pymatgenで特定結晶の空間群調査**

**問題** : pymatgenを使って、ダイヤモンド（C）の結晶構造の空間群を調べてください。ダイヤモンドは以下の特徴を持ちます：

  * 格子定数: $a = 3.567$ Å（立方晶）
  * 原子座標: (0, 0, 0), (0.25, 0.25, 0.25), (0.5, 0.5, 0), (0.75, 0.75, 0.25), (0.5, 0, 0.5), (0.75, 0.25, 0.75), (0, 0.5, 0.5), (0.25, 0.75, 0.75)

**解答コード** :
    
    
    from pymatgen.core import Structure, Lattice
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
    
    # ダイヤモンド構造
    lattice_diamond = Lattice.cubic(3.567)
    
    diamond_structure = Structure(
        lattice_diamond,
        ["C"] * 8,
        [
            [0, 0, 0], [0.25, 0.25, 0.25],
            [0.5, 0.5, 0], [0.75, 0.75, 0.25],
            [0.5, 0, 0.5], [0.75, 0.25, 0.75],
            [0, 0.5, 0.5], [0.25, 0.75, 0.75]
        ]
    )
    
    sga = SpacegroupAnalyzer(diamond_structure)
    
    print("【ダイヤモンド（C）の結晶構造解析】")
    print(f"空間群番号: {sga.get_space_group_number()}")
    print(f"空間群記号: {sga.get_space_group_symbol()}")
    print(f"点群: {sga.get_point_group_symbol()}")
    print(f"結晶系: {sga.get_crystal_system()}")
    print(f"格子タイプ: {sga.get_lattice_type()}")

**期待される出力** :
    
    
    【ダイヤモンド（C）の結晶構造解析】
    空間群番号: 227
    空間群記号: Fd-3m
    点群: m-3m
    結晶系: cubic
    格子タイプ: face_centered

**解説** :

  * ダイヤモンドは、空間群227番 Fd-3m（face-centered diamond cubic）
  * FCC格子に2原子基底を配置した構造（diamond cubic structure）
  * Si, Geもこの構造を持つ

* * *

## 2.7 本章のまとめ

### 学んだこと

  1. **14種類のブラベー格子**
     * 7つの結晶系と4つの配置タイプ（P, C, I, F）の組み合わせ
     * 立方晶系の3種（P, I, F）が材料科学で最重要
     * 六方晶系（HCP）と立方晶系（FCC, BCC）が金属材料の主要構造
  2. **対称操作**
     * 点対称操作: 回転（1, 2, 3, 4, 6）、鏡映（m）、反転（$\bar{1}$）、回反（$\bar{2}, \bar{3}, \bar{4}, \bar{6}$）
     * 並進対称操作: らせん（screw）、映進（glide）
     * 結晶学的制限: 5回や7回以上の回転対称は存在しない
  3. **空間群**
     * 230種類の空間群がすべての結晶構造を分類
     * 空間群 = 点群（32種類） + 並進対称操作
     * Hermann-Mauguin記号（国際記号）で表記
     * 一般位置と特殊位置（Wyckoff position）
  4. **pymatgenによる実践**
     * 空間群情報の取得と解析
     * 結晶構造からの空間群自動判定
     * 対称操作による等価位置の生成
     * Materials Project APIとの連携（材料検索）

### 重要なポイント

  * ブラベー格子は、結晶の「骨組み」を記述する
  * 空間群は、結晶の「完全な対称性」を記述する
  * 同じブラベー格子でも、原子の配置（基底）により異なる空間群になる
  * FCC（225番）、BCC（229番）、HCP（194番）は最も重要な空間群
  * pymatgenは、結晶学解析の強力なツール

### 次の章へ

第3章では、**ミラー指数と逆格子** を学びます：

  * ミラー指数による結晶面・方向の表記
  * 面間隔の計算
  * 逆格子の概念とX線回折への応用
  * Ewald球とBragg条件
  * pymatgenによる逆格子ベクトルの計算
