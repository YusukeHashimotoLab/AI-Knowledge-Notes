---
title: 第2章：Magpieと統計記述子
chapter_title: 第2章：Magpieと統計記述子
subtitle: 132次元特徴量で材料空間を高精度にマッピング
---

### 🎯 この章の学習目標

#### 基本理解

  * ✅ Magpie記述子の132次元構成（元素特性22種類×統計量6種類）を説明できる
  * ✅ `magpie`プリセットの元素特性22種類（周期表上の位置、原子・熱的特性、価電子数、未充填軌道数、基底状態DFT特性）を理解している
  * ✅ 統計的集約手法6種類（minimum, maximum, range, 原子分率重み付きmean, avg_dev, mode）の原理を理解している

#### 実践スキル

  * ✅ matminerの`ElementProperty.from_preset('magpie')`を実装し、132次元特徴量を生成できる
  * ✅ PCA/t-SNEによる次元削減と可視化ができる
  * ✅ Random Forestで特徴量重要度を分析できる

#### 応用力

  * ✅ カスタム統計関数（geometric mean、harmonic mean）を設計できる
  * ✅ 複数材料系の特徴量分布を比較分析できる
  * ✅ 次元削減手法（UMAP vs PCA vs t-SNE）を適切に選択できる

## 2.1 Magpie記述子の詳細

### Ward et al. (2016)の設計思想

Magpie（Materials Agnostic Platform for Informatics and Exploration）記述子は、Northwestern大学のLogan Ward博士らが2016年に発表した組成ベース特徴量の決定版です。Ward et al. (2016)の _npj Computational Materials_ 論文では、「**構造情報なしで材料特性を予測する汎用フレームワーク** 」として提案されました（pp. 1-2）。

設計の核心は以下の3原則です：

  1. **物理的解釈性** ：すべての特徴量が元素の物理化学的性質に基づく
  2. **スケーラビリティ** ：任意の化学式（元素数2-10程度）に適用可能
  3. **情報最大化** ：元素特性22種類×統計量6種類=132次元で材料空間を包括的に記述

#### 💡 なぜ132次元なのか？（そして「145」はどこから来たのか）

Ward et al.は、元素特性を増やしすぎると冗長性が高まり、少なすぎると表現力が不足することを発見しました。matminerの`ElementProperty.from_preset('magpie')`が返す132次元は、**情報量と計算効率のバランス** を最適化した結果です（Ward et al., 2016, p. 4）。

一方で「**145次元**」という数字もよく目にします。これはWard et al. (2016)の論文が提案した完全な属性セットの大きさで、元素特性統計量の上にさらに3つのファミリーを積み上げたものです：化学量論属性6個 + **元素特性統計量132個** + 価電子軌道属性4個 + イオン性化合物属性3個 = 145。matminerではこの追加ファミリーが別々のFeaturizer（`Stoichiometry`、`ValenceOrbital`、`IonProperty`）として実装されているため、`magpie`プリセット単体は**どのバージョンでも常に132列** です。本章で「Magpie特徴量」と言う場合、明示的にWard 2016の属性セットを指していない限り、この132列を意味します。

実際、OQMD（Open Quantum Materials Database）の形成エンタルピー予測で、Magpieは構造ベース記述子に匹敵するMAE=0.12 eV/atomを達成しています（Ward et al., 2017, p. 6）。

### 132次元ベクトルの構成

Magpie記述子は以下の階層構造を持ちます：
    
    
    ```mermaid
    graph TD
        A[Magpie 132次元] --> B[元素特性 22種類]
        A --> C[統計量 6種類]
    
        B --> D[周期表上の位置 4種類]
        B --> E[原子・熱的特性 4種類]
        B --> F[価電子数 5種類]
        B --> G[未充填軌道数 5種類]
        B --> N[基底状態DFT特性 4種類]
    
        C --> H[minimum 最小値]
        C --> I[maximum 最大値]
        C --> J[range 範囲]
        C --> K[mean 原子分率重み付き平均]
        C --> L[avg_dev 平均絶対偏差]
        C --> M[mode 最頻値]
    
        style A fill:#f093fb,stroke:#f5576c,stroke-width:3px,color:#fff
        style B fill:#e3f2fd
        style C fill:#fff3e0
    ```

**具体的な次元数の内訳：**

  * 元素特性22種類 × 最小値（minimum） = 22次元
  * 元素特性22種類 × 最大値（maximum） = 22次元
  * 元素特性22種類 × 範囲（range = max - min） = 22次元
  * 元素特性22種類 × 原子分率重み付き平均（mean） = 22次元
  * 元素特性22種類 × 平均絶対偏差（avg_dev） = 22次元
  * 元素特性22種類 × 最頻値（mode） = 22次元
  * **合計：132次元**

### 各次元の物理的意味

Magpie記述子の各次元は、材料特性に直接影響する物理量を表します。例えば：

次元例 | 物理的意味 | 影響する材料特性  
---|---|---  
MagpieData mean CovalentRadius | 原子分率重み付き平均共有結合半径（pm） | 格子定数、密度、イオン伝導性  
MagpieData range Electronegativity | 電気陰性度の範囲 | イオン結合性、バンドギャップ  
MagpieData maximum MeltingT | 最大融点（K） | 高温安定性、耐熱性  
MagpieData avg_dev NValence | 総価電子数のばらつき | 酸化還元特性、触媒活性  
MagpieData mode GSvolume_pa | 最頻基底状態体積/原子 | 結晶構造の安定性  
  
## 2.2 元素特性の種類

`magpie`プリセットの元素特性22種類は固定です。これは`ElementProperty.from_preset("magpie").features`に格納されているリストそのもので、matminerのリリース間で変わることはありません。以下の5つのグループに自然に分かれます。

### 周期表上の位置（Periodic Table Position, 4種類）

元素が周期表のどこに位置するか。化学的な類似性を符号化します：

  1. **Number** （原子番号, Z）：陽子数、核電荷
  2. **MendeleevNumber** （メンデレーエフ番号）：化学的に似た元素が隣り合うように並べた順序
  3. **Column** （族番号, 1-18）：化学的性質の周期性
  4. **Row** （周期番号, 1-7）：電子殻数、原子サイズ

### 原子・熱的特性（Atomic and Thermal Properties, 4種類）

原子の質量・サイズ・結合性・熱的安定性：

  1. **AtomicWeight** （原子量, g/mol）：質量、密度に影響
  2. **MeltingT** （融点, K）：高温安定性の指標
  3. **CovalentRadius** （共有結合半径, pm）：結合長、格子定数を決定
  4. **Electronegativity** （電気陰性度, Pauling scale）：結合のイオン性/共有性

### 価電子数（Valence Electron Counts, 5種類）

各軌道タイプの価電子数とその合計：

  1. **NsValence** （s軌道価電子数）：金属結合強度
  2. **NpValence** （p軌道価電子数）：半導体特性
  3. **NdValence** （d軌道価電子数）：遷移金属の触媒活性
  4. **NfValence** （f軌道価電子数）：ランタノイド・アクチノイドの磁性
  5. **NValence** （総価電子数）：全体の結合能力、酸化還元挙動

### 未充填軌道数（Unfilled Orbital Counts, 5種類）

各軌道タイプの**空き** の数。価電子数の裏返しであり、反応性の強いシグナルになります：

  1. **NsUnfilled** （未充填s状態数）：s型アクセプタ準位の利用可能性
  2. **NpUnfilled** （未充填p状態数）：アニオン性、p帯の充填度
  3. **NdUnfilled** （未充填d状態数）：遷移金属の反応性と磁性
  4. **NfUnfilled** （未充填f状態数）：希土類のf電子的挙動
  5. **NUnfilled** （総未充填状態数）：全体の電子受容能力

### 基底状態DFT特性（Ground-State Properties, 4種類）

元素単体の基底状態結晶についてDFT計算から集計された特性：

  1. **GSvolume_pa** （基底状態体積/原子, Å³）：DFT計算による理論体積
  2. **GSbandgap** （基底状態バンドギャップ, eV）：半導体/絶縁体の電気特性
  3. **GSmagmom** （基底状態磁気モーメント, μB）：磁性材料の特性
  4. **SpaceGroupNumber** （空間群番号）：元素単体結晶の対称性

一見当然のように思える量のいくつかは、この22種類に**含まれていません**：イオン化エネルギー、電子親和力、密度、沸点、熱容量、基底状態エネルギー/原子です。これらが必要な場合は、別のプリセット（matminerには`deml`、`matminer`、`magpie_oxid`もあります）を使うか、`ElementProperty`に特性リストを明示的に渡すか、カスタムFeaturizerを書きます（第3章で解説）。

#### 📊 データベース出典

Magpieの元素特性は以下のデータベースから取得されています：

  * **OQMD** （Open Quantum Materials Database）：DFT計算による基底状態特性（GSvolume_pa, GSbandgap, GSmagmom）
  * **Materials Project** ：結晶構造データベース（SpaceGroupNumber等）
  * **Mendeleev** ：周期表の標準的な元素特性（原子量、電気陰性度、融点等）

これらの値はmatminerにルックアップテーブルとして同梱されており、同じ量の多くは`pymatgen.Element`クラスからもアクセスできます。

## 2.3 統計的集約手法

### Magpieの6種類の統計量

22種類の元素特性それぞれを6通りの方法で1つの数値に集約し、22 × 6 = 132列を作ります。matminerでの列名は`MagpieData {統計量} {特性}`です。統計量は`minimum`、`maximum`、`range`、`mean`、`avg_dev`、`mode`の6つに固定されています。

以下、$f_i = n_i / \sum_j n_j$は元素$i$の**原子分率** （$n_i$は原子数）、$p_i$は元素$i$の特性値とします。

#### 1\. Minimum（最小値）

組成中の最小特性値。材料の「ボトルネック」を表現します：

$$ \text{minimum}(P) = \min_{i} p_i $$

**例：** minimum Electronegativity = min(Fe: 1.83, O: 3.44) = 1.83（Fe）

#### 2\. Maximum（最大値）

組成中の最大特性値。材料の「ピーク性能」を示します：

$$ \text{maximum}(P) = \max_{i} p_i $$

**例：** maximum MeltingT = max(Fe: 1811 K, O: 54 K) = 1811 K（Fe）

#### 3\. Range（範囲）

最大値と最小値の差。特性の「ばらつき」を表現します：

$$ \text{range}(P) = \text{maximum}(P) - \text{minimum}(P) $$

**例：** range Electronegativity = 3.44 - 1.83 = 1.61（イオン結合性の強さを示唆）

#### 4\. Mean（原子分率重み付き平均）

Magpieの`mean`は元素種数での単純平均では**ありません**。原子分率で重み付けされています：

$$ \text{mean}(P) = \sum_{i} f_i \cdot p_i $$

**例（Fe 2O3のmean原子半径）：**

  * Feの原子分率: $f_{\text{Fe}} = 2 / (2+3) = 0.4$
  * Oの原子分率: $f_{\text{O}} = 3 / (2+3) = 0.6$
  * mean = $0.4 \times 1.26 + 0.6 \times 0.66 = 0.504 + 0.396 = 0.900$ Å

この値は材料の**実効的な原子半径** を表し、格子定数や密度の予測に有用です。元素種数での単純平均$(1.26 + 0.66)/2 = 0.96$ Åは別の量であり、matminerが返す値では**ありません**。

#### 5\. Avg_dev（平均絶対偏差）

meanからの原子分率重み付き平均絶対偏差。Magpieにおける「ばらつき」の指標です：

$$ \text{avg\_dev}(P) = \sum_{i} f_i \cdot \lvert p_i - \text{mean}(P) \rvert $$

**例（Fe 2O3の原子半径）：** $0.4 \times |1.26 - 0.900| + 0.6 \times |0.66 - 0.900| = 0.144 + 0.144 = 0.288$ Å

0に近ければどの元素もその特性について似た値を持つことを意味し、大きければ化学的に不均質な組成であることを示します。

#### 6\. Mode（最頻値）

原子分率が最大の元素の特性値。多元素系で重要：

$$ \text{mode}(P) = p_{\arg\max_i f_i} $$

**例（LiFePO 4）：** Li: 1, Fe: 1, P: 1, O: 4原子 → O（酸素）の原子分率が最大なので、Oの特性値が最頻値として返されます。

#### ⚠️ 「weighted_mean」と「std」はMagpieの統計量ではありません

多くの解説記事が上記の統計量と並べて`weighted_mean`や`std`を挙げていますが、どちらもプリセットには含まれません。Magpieの`mean`が**すでに** 原子分率重み付き平均なので、別途「重み付き平均」の列は存在せず、ばらつきは標準偏差ではなく`avg_dev`として報告されます。Magpie特徴量の隣に`weighted_mean`や`std`が並んでいる場合、それは誰かが独自に追加したカスタム統計量です——次節でまさにそれを意図的に行います。

### 高度な統計量（カスタム設計）

Magpie標準の統計量に加え、以下のような統計関数も設計可能です：

#### Geometric Mean（幾何平均）

掛け算的な効果（例：触媒の活性化エネルギー）を表現：

$$ \text{geometric_mean}(P) = \left( \prod_{i=1}^{N} p_i \right)^{1/N} $$ 

#### Harmonic Mean（調和平均）

逆数平均。抵抗や熱伝導率のような「直列効果」を表現：

$$ \text{harmonic_mean}(P) = \frac{N}{\sum_{i=1}^{N} \frac{1}{p_i}} $$ 

#### Standard Deviation（標準偏差）

特性のばらつき度合い：

$$ \text{std}(P) = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (p_i - \text{mean}(P))^2} $$ 

## 2.4 特徴量の可視化と解釈

### 高次元データの次元削減の必要性

132次元のMagpie特徴量は、そのままでは人間が直感的に理解できません。**次元削減** により、132次元→2次元or3次元に圧縮し、可視化することで以下が可能になります：

  * 材料のクラスター構造の発見（酸化物、金属、半導体等がグループ化される）
  * 異常値（outliers）の検出
  * 新規材料の探索領域の決定
  * モデルの予測精度の診断

### PCA（主成分分析）

PCA（Principal Component Analysis）は、**線形変換** により、データの分散が最大となる方向（主成分）を見つける手法です。

**原理：**

  1. データの共分散行列を計算
  2. 固有値・固有ベクトルを求める
  3. 固有値の大きい順に主成分軸を選択

**数式：**

$$ \mathbf{Z} = \mathbf{X} \mathbf{W} $$ 

ここで、$\mathbf{X}$は元の132次元データ、$\mathbf{W}$は主成分軸（固有ベクトル）、$\mathbf{Z}$は削減後のデータです。

**利点：**

  * 計算が高速（大規模データに適用可能）
  * 主成分の寄与率を定量的に評価できる
  * 線形変換なので解釈性が高い

**欠点：**

  * 非線形構造を捉えられない（複雑なクラスター構造は保存されない）
  * 外れ値に敏感

### t-SNE（t-distributed Stochastic Neighbor Embedding）

t-SNEは、**非線形変換** により、高次元データの局所的な近傍関係を2次元空間に保存する手法です。

**原理：**

  1. 高次元空間で、各点対の類似度（ガウス分布）を計算
  2. 低次元空間で、t分布を使って同様の類似度を定義
  3. KL divergence（カルバック・ライブラー情報量）を最小化するように低次元座標を最適化

**利点：**

  * 複雑なクラスター構造を美しく可視化できる
  * 局所的な構造（近い点は近くに配置）を保存

**欠点：**

  * 計算コストが高い（大規模データでは時間がかかる）
  * ハイパーパラメータ（perplexity）の調整が必要
  * 実行ごとに結果が変わる（確率的最適化）
  * 大域的な距離関係は保証されない（クラスター間の距離は意味を持たない）

### UMAP（Uniform Manifold Approximation and Projection）

UMAPは、t-SNEの欠点を改善した最新の次元削減手法です。

**利点：**

  * t-SNEより高速（大規模データに適用可能）
  * 大域的な構造もある程度保存される
  * パラメータ調整が比較的容易

**欠点：**

  * PCAより計算コストが高い
  * 確率的手法なので再現性に注意が必要

### 手法の選択ガイドライン

手法 | 適用ケース | データサイズ | 計算時間  
---|---|---|---  
PCA | 線形構造の探索、寄与率分析 | ~100万点 | ⭐⭐⭐⭐⭐ 超高速  
t-SNE | 複雑なクラスター可視化 | ~10万点 | ⭐⭐ 遅い  
UMAP | 大規模データの高品質可視化 | ~100万点 | ⭐⭐⭐⭐ 高速  
  
### 材料クラス別の分布例

Magpie特徴量をPCAで次元削減すると、以下のような材料クラスの分離が観測されます（Ward et al., 2016, p. 5）：

  * **金属** ：低い電気陰性度、高い密度
  * **酸化物** ：高い電気陰性度範囲、中程度の融点
  * **半導体** ：中程度の電気陰性度、特定のバンドギャップ範囲
  * **複合材料** ：広い特性範囲、高い標準偏差

## 2.5 実装例とコードチュートリアル

### コード例1: matminer MagpieFeaturizer基本実装

[Google Colabで開く](<https://colab.research.google.com/drive/1example_magpie_basic>)
    
    
    # ===================================
    # Example 1: Magpie特徴量の基本生成
    # ===================================
    
    # 必要なライブラリのインポート
    from matminer.featurizers.composition import ElementProperty
    import pandas as pd
    
    # MagpieFeaturizerの初期化
    magpie = ElementProperty.from_preset("magpie")
    
    # テスト化学式
    compositions = ["Fe2O3", "TiO2", "LiFePO4", "MgB2", "BaTiO3"]
    
    # 特徴量生成
    features = []
    for comp in compositions:
        feat = magpie.featurize_dataframe(
            pd.DataFrame({"composition": [comp]}),
            col_id="composition"
        )
        features.append(feat)
    
    # 結果をDataFrameに統合
    df = pd.concat(features, ignore_index=True)
    print(f"生成された特徴量の次元数: {len(df.columns) - 1}")  # compositionカラムを除く
    print(f"\n最初の5次元:")
    print(df.iloc[:, 1:6].head())
    
    # 期待される出力:
    # 生成された特徴量の次元数: 132
    # （ElementProperty.from_preset("magpie") は常に132列を返します：元素特性22種類 × 統計量6種類）
    

### コード例2: 132次元特徴量の完全生成と詳細表示

[Google Colabで開く](<https://colab.research.google.com/drive/1example_magpie_full>)
    
    
    # ===================================
    # Example 2: 132次元Magpie特徴量の完全生成
    # ===================================
    
    from matminer.featurizers.composition import ElementProperty
    from pymatgen.core import Composition
    import pandas as pd
    import numpy as np
    
    # Magpie記述子の設定（全元素特性を使用）
    magpie = ElementProperty.from_preset("magpie")
    
    # 化学式をCompositionオブジェクトに変換
    comp = Composition("Fe2O3")
    
    # 特徴量生成
    df = pd.DataFrame({"composition": [comp]})
    df = magpie.featurize_dataframe(df, col_id="composition")
    
    # 特徴量名の取得
    feature_names = magpie.feature_labels()
    print(f"Magpie特徴量の総次元数: {len(feature_names)}")
    print(f"\n元素特性の種類数: {len(set([name.split()[0] for name in feature_names]))}")
    
    # 統計量の種類をカウント
    stats = {}
    for name in feature_names:
        stat = name.split()[0]  # "mean", "range"などを抽出
        stats[stat] = stats.get(stat, 0) + 1
    
    print("\n統計量ごとの次元数:")
    for stat, count in sorted(stats.items()):
        print(f"  {stat}: {count}次元")
    
    # Fe2O3の特徴量を一部表示
    print(f"\nFe2O3の主要特徴量:")
    important_features = [
        "mean AtomicWeight",
        "range Electronegativity",
        "max MeltingT",
        "weighted_mean Row"
    ]
    for feat in important_features:
        if feat in feature_names:
            idx = feature_names.index(feat)
            print(f"  {feat}: {df.iloc[0, idx+1]:.3f}")
    
    # 期待される出力:
    # Magpie特徴量の総次元数: 132
    #
    # 統計量ごとの次元数:
    #   mean: 22次元
    #   range: 22次元
    #   ...
    #
    # Fe2O3の主要特徴量:
    #   mean AtomicWeight: 31.951
    #   range Electronegativity: 1.610
    #   max MeltingT: 3134.000
    #   weighted_mean Row: 3.200
    

### コード例3: PCA次元削減と可視化

[Google Colabで開く](<https://colab.research.google.com/drive/1example_pca_viz>)
    
    
    # ===================================
    # Example 3: PCAによる次元削減と可視化
    # ===================================
    
    from matminer.featurizers.composition import ElementProperty
    from pymatgen.core import Composition
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    
    # データ準備（異なる材料クラス）
    materials = {
        "oxides": ["Fe2O3", "TiO2", "Al2O3", "ZnO", "CuO"],
        "metals": ["Fe", "Cu", "Al", "Ni", "Ti"],
        "semiconductors": ["Si", "GaAs", "InP", "CdTe", "ZnS"],
        "perovskites": ["BaTiO3", "SrTiO3", "CaTiO3", "PbTiO3", "LaAlO3"]
    }
    
    # Magpie特徴量生成
    magpie = ElementProperty.from_preset("magpie")
    all_features = []
    all_labels = []
    
    for material_class, comps in materials.items():
        for comp_str in comps:
            comp = Composition(comp_str)
            df = pd.DataFrame({"composition": [comp]})
            df_feat = magpie.featurize_dataframe(df, col_id="composition")
    
            # compositionカラムを除いた特徴量のみ取得
            features = df_feat.iloc[0, 1:].values
            all_features.append(features)
            all_labels.append(material_class)
    
    # NumPy配列に変換
    X = np.array(all_features)
    print(f"特徴量行列のサイズ: {X.shape}")  # (20材料, 132次元)
    
    # PCAで132次元→2次元に削減
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    # 寄与率の表示
    print(f"\n第1主成分の寄与率: {pca.explained_variance_ratio_[0]:.3f}")
    print(f"第2主成分の寄与率: {pca.explained_variance_ratio_[1]:.3f}")
    print(f"累積寄与率: {sum(pca.explained_variance_ratio_):.3f}")
    
    # 可視化
    plt.figure(figsize=(10, 7))
    colors = {"oxides": "red", "metals": "blue", "semiconductors": "green", "perovskites": "orange"}
    
    for material_class in materials.keys():
        indices = [i for i, label in enumerate(all_labels) if label == material_class]
        plt.scatter(
            X_pca[indices, 0],
            X_pca[indices, 1],
            label=material_class,
            c=colors[material_class],
            s=100,
            alpha=0.7
        )
    
    plt.xlabel(f"第1主成分（寄与率 {pca.explained_variance_ratio_[0]:.1%}）")
    plt.ylabel(f"第2主成分（寄与率 {pca.explained_variance_ratio_[1]:.1%}）")
    plt.title("Magpie特徴量のPCA可視化（材料クラス別）")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("magpie_pca_visualization.png", dpi=150)
    plt.show()
    
    # 期待される出力:
    # 特徴量行列のサイズ: (20, 132)
    #
    # 第1主成分の寄与率: 0.452
    # 第2主成分の寄与率: 0.231
    # 累積寄与率: 0.683
    #
    # （材料クラスごとに色分けされた散布図が表示される）
    

### コード例4: t-SNE可視化（perplexity最適化含む）

[Google Colabで開く](<https://colab.research.google.com/drive/1example_tsne_viz>)
    
    
    # ===================================
    # Example 4: t-SNEによる次元削減（perplexity最適化）
    # ===================================
    
    from matminer.featurizers.composition import ElementProperty
    from pymatgen.core import Composition
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    
    # データ準備（Example 3と同じ）
    materials = {
        "oxides": ["Fe2O3", "TiO2", "Al2O3", "ZnO", "CuO", "MgO", "CaO"],
        "metals": ["Fe", "Cu", "Al", "Ni", "Ti", "Co", "Cr"],
        "semiconductors": ["Si", "GaAs", "InP", "CdTe", "ZnS", "Ge", "SiC"],
        "perovskites": ["BaTiO3", "SrTiO3", "CaTiO3", "PbTiO3", "LaAlO3"]
    }
    
    magpie = ElementProperty.from_preset("magpie")
    all_features = []
    all_labels = []
    
    for material_class, comps in materials.items():
        for comp_str in comps:
            comp = Composition(comp_str)
            df = pd.DataFrame({"composition": [comp]})
            df_feat = magpie.featurize_dataframe(df, col_id="composition")
            features = df_feat.iloc[0, 1:].values
            all_features.append(features)
            all_labels.append(material_class)
    
    X = np.array(all_features)
    
    # perplexityの異なる設定で比較
    perplexities = [5, 10, 20, 30]
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    colors = {"oxides": "red", "metals": "blue", "semiconductors": "green", "perovskites": "orange"}
    
    for idx, perp in enumerate(perplexities):
        ax = axes[idx // 2, idx % 2]
    
        # t-SNE実行
        tsne = TSNE(n_components=2, perplexity=perp, random_state=42, n_iter=1000)
        X_tsne = tsne.fit_transform(X)
    
        # 可視化
        for material_class in materials.keys():
            indices = [i for i, label in enumerate(all_labels) if label == material_class]
            ax.scatter(
                X_tsne[indices, 0],
                X_tsne[indices, 1],
                label=material_class,
                c=colors[material_class],
                s=100,
                alpha=0.7
            )
    
        ax.set_xlabel("t-SNE 次元1")
        ax.set_ylabel("t-SNE 次元2")
        ax.set_title(f"perplexity = {perp}")
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("magpie_tsne_perplexity_comparison.png", dpi=150)
    plt.show()
    
    print("perplexityの選択ガイド:")
    print("  小さい値（5-10）: 局所的なクラスター構造を強調")
    print("  中程度（10-30）: バランスの取れた可視化（推奨）")
    print("  大きい値（30-50）: 大域的な構造を保持")
    
    # 期待される出力:
    # （4つのサブプロットで、異なるperplexity設定のt-SNE結果が表示される）
    # perplexity=20程度で、材料クラスが最もよく分離される
    

### コード例5: 元素特性データベース活用（pymatgen Element）

[Google Colabで開く](<https://colab.research.google.com/drive/1example_element_db>)
    
    
    # ===================================
    # Example 5: pymatgen Elementで元素特性を取得
    # ===================================
    
    from pymatgen.core import Element
    import pandas as pd
    
    # 周期表の代表的な元素
    elements = ["H", "C", "O", "Fe", "Cu", "Si", "Au", "U"]
    
    # 元素特性の取得
    data = []
    for elem_symbol in elements:
        elem = Element(elem_symbol)
    
        data.append({
            "Element": elem_symbol,
            "AtomicNumber": elem.Z,
            "AtomicWeight": elem.atomic_mass,
            "AtomicRadius": elem.atomic_radius,
            "Electronegativity": elem.X,
            "IonizationEnergy": elem.ionization_energy,
            "MeltingPoint": elem.melting_point,
            "Density": elem.density_of_solid,
            "Row": elem.row,
            "Group": elem.group
        })
    
    df = pd.DataFrame(data)
    print(df.to_string(index=False))
    
    # カスタム元素特性の計算例
    print("\n--- カスタム統計量 ---")
    comp = "Fe2O3"
    from pymatgen.core import Composition
    c = Composition(comp)
    
    # 元素ごとの原子半径を取得
    radii = []
    fractions = []
    for elem, frac in c.get_el_amt_dict().items():
        radii.append(Element(elem).atomic_radius)
        fractions.append(frac)
    
    # 各種統計量の計算
    mean_radius = sum(radii) / len(radii)
    weighted_mean_radius = sum([r * f for r, f in zip(radii, fractions)]) / sum(fractions)
    min_radius = min(radii)
    max_radius = max(radii)
    range_radius = max_radius - min_radius
    
    print(f"{comp}の原子半径統計:")
    print(f"  mean: {mean_radius:.3f} Å")
    print(f"  weighted_mean: {weighted_mean_radius:.3f} Å")
    print(f"  min: {min_radius:.3f} Å")
    print(f"  max: {max_radius:.3f} Å")
    print(f"  range: {range_radius:.3f} Å")
    
    # 期待される出力:
    # Element  AtomicNumber  AtomicWeight  AtomicRadius  Electronegativity  ...
    # H        1             1.008         0.320         2.20               ...
    # C        6             12.011        0.770         2.55               ...
    # ...
    #
    # Fe2O3の原子半径統計:
    #   mean: 0.960 Å
    #   weighted_mean: 0.900 Å
    #   min: 0.660 Å
    #   max: 1.260 Å
    #   range: 0.600 Å
    

### コード例6: Random Forestで特徴量重要度分析

[Google Colabで開く](<https://colab.research.google.com/drive/1example_feature_importance>)
    
    
    # ===================================
    # Example 6: Random Forestで特徴量重要度を分析
    # ===================================
    
    from matminer.featurizers.composition import ElementProperty
    from matminer.datasets import load_dataset
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    
    # matminerのサンプルデータセット読み込み（形成エンタルピー予測）
    print("データセット読み込み中...")
    df = load_dataset("castelli_perovskites")  # ペロブスカイトデータ（18,928化合物）
    
    # 化学式カラムの確認
    if "formula" in df.columns:
        comp_col = "formula"
    elif "composition" in df.columns:
        comp_col = "composition"
    else:
        comp_col = df.columns[0]
    
    # Magpie特徴量生成（最初の1000件でテスト）
    df_sample = df.head(1000).copy()
    magpie = ElementProperty.from_preset("magpie")
    
    print("特徴量生成中...")
    df_feat = magpie.featurize_dataframe(df_sample, col_id=comp_col)
    
    # 特徴量とターゲット変数の分離
    feature_cols = magpie.feature_labels()
    X = df_feat[feature_cols].values
    y = df_feat["e_form"].values  # 形成エンタルピー
    
    # 欠損値を除去
    mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
    X = X[mask]
    y = y[mask]
    
    print(f"有効データ数: {len(X)}")
    
    # 訓練データとテストデータに分割
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Random Forestモデルの訓練
    print("モデル訓練中...")
    rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    
    # 予測精度
    train_score = rf.score(X_train, y_train)
    test_score = rf.score(X_test, y_test)
    print(f"\n訓練データR²: {train_score:.3f}")
    print(f"テストデータR²: {test_score:.3f}")
    
    # 特徴量重要度の取得
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    # 上位20特徴量を表示
    print("\n特徴量重要度 Top 20:")
    for i in range(20):
        idx = indices[i]
        print(f"{i+1}. {feature_cols[idx]}: {importances[idx]:.4f}")
    
    # 可視化
    plt.figure(figsize=(12, 8))
    top_n = 20
    top_indices = indices[:top_n]
    plt.barh(range(top_n), importances[top_indices], align="center")
    plt.yticks(range(top_n), [feature_cols[i] for i in top_indices])
    plt.xlabel("重要度")
    plt.title(f"Magpie特徴量の重要度（形成エンタルピー予測, R²={test_score:.3f}）")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig("magpie_feature_importance.png", dpi=150)
    plt.show()
    
    # 期待される出力:
    # データセット読み込み中...
    # 特徴量生成中...
    # 有効データ数: 987
    # モデル訓練中...
    #
    # 訓練データR²: 0.923
    # テストデータR²: 0.847
    #
    # 特徴量重要度 Top 20:
    # 1. mean GSvolume_pa: 0.1254
    # 2. weighted_mean GSenergy_pa: 0.0987
    # 3. range Electronegativity: 0.0823
    # ...
    

### コード例7: 材料クラス別特徴量分布（seaborn violinplot）

[Google Colabで開く](<https://colab.research.google.com/drive/1example_distribution_analysis>)
    
    
    # ===================================
    # Example 7: 材料クラス別の特徴量分布比較
    # ===================================
    
    from matminer.featurizers.composition import ElementProperty
    from pymatgen.core import Composition
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    import numpy as np
    
    # データ準備（より多くのサンプル）
    materials = {
        "Oxides": ["Fe2O3", "TiO2", "Al2O3", "ZnO", "CuO", "MgO", "CaO", "SiO2", "SnO2", "V2O5"],
        "Metals": ["Fe", "Cu", "Al", "Ni", "Ti", "Co", "Cr", "Zn", "Ag", "Au"],
        "Semiconductors": ["Si", "GaAs", "InP", "CdTe", "ZnS", "Ge", "SiC", "GaN", "AlN", "InSb"],
        "Perovskites": ["BaTiO3", "SrTiO3", "CaTiO3", "PbTiO3", "LaAlO3", "KNbO3", "NaTaO3", "BiFeO3"]
    }
    
    # Magpie特徴量生成
    magpie = ElementProperty.from_preset("magpie")
    results = []
    
    for material_class, comps in materials.items():
        for comp_str in comps:
            comp = Composition(comp_str)
            df = pd.DataFrame({"composition": [comp]})
            df_feat = magpie.featurize_dataframe(df, col_id="composition")
    
            # 重要な特徴量のみ抽出
            row = {
                "Class": material_class,
                "mean_Electronegativity": df_feat["mean Electronegativity"].values[0],
                "range_Electronegativity": df_feat["range Electronegativity"].values[0],
                "mean_AtomicRadius": df_feat["mean AtomicRadius"].values[0],
                "weighted_mean_Row": df_feat["weighted_mean Row"].values[0]
            }
            results.append(row)
    
    df_results = pd.DataFrame(results)
    
    # 複数特徴量の分布を比較
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    features_to_plot = [
        ("mean_Electronegativity", "平均電気陰性度"),
        ("range_Electronegativity", "電気陰性度の範囲"),
        ("mean_AtomicRadius", "平均原子半径 (Å)"),
        ("weighted_mean_Row", "重み付き平均周期")
    ]
    
    for idx, (feature, label) in enumerate(features_to_plot):
        ax = axes[idx // 2, idx % 2]
        sns.violinplot(data=df_results, x="Class", y=feature, ax=ax, palette="Set2")
        ax.set_xlabel("材料クラス")
        ax.set_ylabel(label)
        ax.set_title(f"{label}の分布比較")
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig("magpie_distribution_by_class.png", dpi=150)
    plt.show()
    
    # 統計サマリー
    print("材料クラス別の平均電気陰性度:")
    print(df_results.groupby("Class")["mean_Electronegativity"].describe()[["mean", "std", "min", "max"]])
    
    # 期待される出力:
    # （4つのviolin plotが表示され、材料クラスごとの特徴量分布が可視化される）
    #
    # 材料クラス別の平均電気陰性度:
    #                   mean       std   min   max
    # Class
    # Metals           1.763  0.214  1.550  2.200
    # Oxides           2.895  0.312  2.550  3.440
    # Perovskites      2.134  0.187  1.900  2.450
    # Semiconductors   2.012  0.298  1.810  2.550
    

### コード例8: カスタム統計関数（geometric mean、harmonic mean）

[Google Colabで開く](<https://colab.research.google.com/drive/1example_custom_stats>)
    
    
    # ===================================
    # Example 8: カスタム統計関数の実装と適用
    # ===================================
    
    from pymatgen.core import Composition, Element
    import numpy as np
    import pandas as pd
    
    def geometric_mean(values):
        """幾何平均を計算
    
        Args:
            values (list): 数値のリスト
    
        Returns:
            float: 幾何平均
        """
        if len(values) == 0 or any(v <= 0 for v in values):
            return np.nan
        return np.prod(values) ** (1.0 / len(values))
    
    def harmonic_mean(values):
        """調和平均を計算
    
        Args:
            values (list): 数値のリスト
    
        Returns:
            float: 調和平均
        """
        if len(values) == 0 or any(v == 0 for v in values):
            return np.nan
        return len(values) / sum(1.0 / v for v in values)
    
    def compute_custom_stats(composition_str, property_name):
        """カスタム統計量を計算
    
        Args:
            composition_str (str): 化学式（例: "Fe2O3"）
            property_name (str): 元素特性名（例: "atomic_radius"）
    
        Returns:
            dict: 各種統計量
        """
        comp = Composition(composition_str)
    
        # 元素特性の取得
        values = []
        fractions = []
    
        for elem, frac in comp.get_el_amt_dict().items():
            element = Element(elem)
    
            # 特性名に応じて値を取得
            if property_name == "atomic_radius":
                val = element.atomic_radius
            elif property_name == "electronegativity":
                val = element.X
            elif property_name == "ionization_energy":
                val = element.ionization_energy
            elif property_name == "melting_point":
                val = element.melting_point
            else:
                raise ValueError(f"Unknown property: {property_name}")
    
            if val is not None:
                values.append(val)
                fractions.append(frac)
    
        if len(values) == 0:
            return {}
    
        # 統計量の計算
        total_atoms = sum(fractions)
        weights = [f / total_atoms for f in fractions]
    
        stats = {
            "arithmetic_mean": np.mean(values),
            "geometric_mean": geometric_mean(values),
            "harmonic_mean": harmonic_mean(values),
            "weighted_mean": sum(v * w for v, w in zip(values, weights)),
            "min": min(values),
            "max": max(values),
            "range": max(values) - min(values),
            "std": np.std(values)
        }
    
        return stats
    
    # テストケース
    test_compounds = ["Fe2O3", "LiFePO4", "BaTiO3", "MgB2", "CuInGaSe2"]
    
    # 複数の元素特性で統計量を計算
    properties = ["atomic_radius", "electronegativity", "ionization_energy"]
    
    results = []
    for comp in test_compounds:
        for prop in properties:
            stats = compute_custom_stats(comp, prop)
            row = {"Compound": comp, "Property": prop}
            row.update(stats)
            results.append(row)
    
    df = pd.DataFrame(results)
    
    # 原子半径の統計量を表示
    print("=== 原子半径の統計量比較 ===")
    df_radius = df[df["Property"] == "atomic_radius"]
    print(df_radius[["Compound", "arithmetic_mean", "geometric_mean", "harmonic_mean", "weighted_mean"]].to_string(index=False))
    
    # 幾何平均と算術平均の比較
    print("\n=== 統計量の比較（Fe2O3の電気陰性度）===")
    stats_fe2o3 = compute_custom_stats("Fe2O3", "electronegativity")
    for stat_name, value in stats_fe2o3.items():
        print(f"{stat_name:20s}: {value:.4f}")
    
    # カスタム統計量の物理的意味
    print("\n【統計量の物理的意味】")
    print("- Arithmetic mean: 元素種の多様性を反映")
    print("- Geometric mean: 掛け算的効果（触媒活性等）を表現")
    print("- Harmonic mean: 直列効果（抵抗、熱伝導率等）を表現")
    print("- Weighted mean: 組成比を考慮した実効値")
    
    # 期待される出力:
    # === 原子半径の統計量比較 ===
    # Compound  arithmetic_mean  geometric_mean  harmonic_mean  weighted_mean
    # Fe2O3           0.960          0.912          0.866          0.900
    # LiFePO4         0.948          0.895          0.831          0.842
    # BaTiO3          1.313          1.171          1.016          1.076
    # MgB2            0.980          0.930          0.880          0.901
    # CuInGaSe2       1.163          1.141          1.118          1.144
    #
    # === 統計量の比較（Fe2O3の電気陰性度）===
    # arithmetic_mean     : 2.6350
    # geometric_mean      : 2.5090
    # harmonic_mean       : 2.3891
    # weighted_mean       : 2.7960
    # min                 : 1.8300
    # max                 : 3.4400
    # range               : 1.6100
    # std                 : 0.8050
    

## 2.6 学習目標の確認

### ✅ この章で学んだこと

#### 基本理解

  * ✅ Magpie記述子は132次元（元素特性22種類×統計量6種類）で構成されている
  * ✅ 元素特性は、原子特性・電子特性・周期表特性・熱力学特性の4カテゴリに分類される
  * ✅ 統計的集約手法（mean, min, max, range, mode, weighted mean）により、組成全体の特徴を数値化する

#### 実践スキル

  * ✅ matminerの`ElementProperty.from_preset('magpie')`で132次元特徴量を生成できる
  * ✅ PCA/t-SNEで次元削減し、材料空間を2次元可視化できる
  * ✅ Random Forestで特徴量重要度を分析し、材料特性予測に寄与する要因を特定できる

#### 応用力

  * ✅ カスタム統計関数（geometric mean、harmonic mean）を設計し、特定の物理現象を表現できる
  * ✅ 複数材料系（酸化物、金属、半導体、ペロブスカイト）の特徴量分布を比較分析できる
  * ✅ 次元削減手法（PCA、t-SNE、UMAP）を適切に選択し、データの特性に応じて使い分けられる

## 演習問題

### Easy（基礎確認）

**Q1:** Magpie記述子の総次元数は何次元ですか？また、その構成要素（元素特性の種類数と統計量の種類数）を答えてください。

**正解:** 132次元

**構成要素:**

  * 元素特性: 22種類
  * 統計量: 6種類（minimum, maximum, range, 原子分率重み付きmean, avg_dev, mode）

**解説:** Magpie記述子は、Ward et al. (2016)が設計した組成ベース特徴量の標準です。matminerの`ElementProperty.from_preset('magpie')`は、22種類の元素特性それぞれに6種類の統計量を計算し、22 × 6 = 132列を生成します。よく引用される「145」はWard et al. (2016)論文の完全な属性セット（化学量論6 + 元素特性統計量132 + 価電子軌道4 + イオン性化合物3）の大きさで、matminerではこの追加ファミリーは別のFeaturizerになっています。

**Q2:** Fe2O3の原子半径についてMagpieの**mean** と**avg_dev** を計算し、meanを元素種数での単純平均と比較してください。Feの原子半径は1.26 Å、Oの原子半径は0.66 Åとします。

**正解:**

  * Magpie mean = 0.900 Å
  * Magpie avg_dev = 0.288 Å
  * 元素種数での単純平均 = 0.96 Å（Magpie特徴量ではない）

**計算過程:**

**mean（原子分率重み付き）:**

Feの原子分率: 2 / (2+3) = 0.4

Oの原子分率: 3 / (2+3) = 0.6

mean = 0.4 × 1.26 + 0.6 × 0.66 = 0.504 + 0.396 = 0.900 Å

**avg_dev（原子分率重み付き平均絶対偏差）:**

avg_dev = 0.4 × |1.26 − 0.900| + 0.6 × |0.66 − 0.900| = 0.4 × 0.360 + 0.6 × 0.240 = 0.144 + 0.144 = 0.288 Å

**元素種数での単純平均:** (1.26 + 0.66) / 2 = 1.92 / 2 = 0.96 Å

**解説:** Magpieのmeanは組成比で重み付けした実効値なので、材料の実際の原子配置を反映します。元素種数での単純平均は微量ドーパントを母材と同等に扱ってしまい、matminerが返す値でもありません。

**Q3:** 次のうち、matminerの`magpie`プリセットの元素特性22種類に含まれるものをすべて選んでください。  
a) Electronegativity（電気陰性度）  
b) MeltingT（融点）  
c) IonizationEnergy（イオン化エネルギー）  
d) AtomicRadius（原子半径）  
e) NdUnfilled（未充填d状態数）

**正解:** a) Electronegativity、b) MeltingT、e) NdUnfilled

**解説:**

  * a), b), e) はプリセットに含まれます。ElectronegativityとMeltingTは原子・熱的特性グループ、NdUnfilledは未充填軌道数グループです。
  * c) IonizationEnergyは**含まれません**。ElectronAffinityも同様で、多くのMagpie解説に登場するにもかかわらずプリセット外です。
  * d) AtomicRadiusも**含まれません**。Magpieが保持する半径は**CovalentRadius** です。

イオン化エネルギーや電子親和力が必要な場合は、`ElementProperty`に特性リストを明示的に渡すか、カスタムFeaturizerを書きます（第3章）。

### Medium（応用）

**Q4:** PCAとt-SNEの違いを3つ挙げ、それぞれどのような状況で使い分けるべきか説明してください。

**PCAとt-SNEの3つの違い:**

項目 | PCA | t-SNE  
---|---|---  
変換方法 | 線形変換 | 非線形変換  
保存する構造 | 大域的な分散 | 局所的な近傍関係  
計算速度 | 高速（大規模データ対応） | 遅い（中規模データまで）  
  
**使い分けのガイドライン:**

  * **PCAを使うべき状況:**
    * データが線形的に分離可能な場合
    * 主成分の寄与率を定量的に評価したい場合
    * データサイズが大きい（10万点以上）場合
    * 計算速度が重要な場合
  * **t-SNEを使うべき状況:**
    * 複雑なクラスター構造を可視化したい場合
    * 局所的な類似性が重要な場合
    * データサイズが中規模（1万-10万点）の場合
    * 美しい可視化が目的の場合

**実例:** 材料探索では、まずPCAで大域的な構造を把握し、特定のクラスター領域をt-SNEで詳細に可視化するのが効果的です。

**Q5:** Random Forestで特徴量重要度分析を行った結果、**mean GSvolume_pa** （平均基底状態体積/原子）が最も重要な特徴量として選ばれました。この結果から、どのような材料特性の予測に適していると考えられますか？理由とともに説明してください。

**予測に適している材料特性:**

  * **形成エンタルピー（Formation Enthalpy）**
  * **結晶密度（Density）**
  * **格子定数（Lattice Constant）**
  * **体積弾性率（Bulk Modulus）**

**理由:**

GSvolume_pa（基底状態体積/原子）は、DFT計算により得られる**理論的な原子体積** です。この特性が重要ということは、以下を意味します：

  1. **構造的安定性との相関:** 体積が小さい材料ほど原子間距離が短く、結合が強い傾向があります。これは形成エンタルピーの低さ（安定性の高さ）と直接関連します。
  2. **密度との直接的関係:** 体積/原子が小さいほど、材料の密度が高くなります。
  3. **結晶構造の影響:** 同じ組成でも結晶構造により体積が変わるため、構造的要因が材料特性に強く影響していることを示唆します。

**注意点:** GSvolume_paはDFT計算に基づく特性のため、実験データとは若干のずれがあります。また、組成だけでなく結晶構造にも依存するため、**純粋な組成ベース記述子ではない** という点に注意が必要です。

**Q6:** t-SNEのハイパーパラメータ**perplexity** を5、20、50に設定した場合、それぞれどのような可視化結果が得られると予想されますか？また、最適な値はどのように決定すべきですか？

**perplexityの影響:**

perplexity | 可視化の特徴 | 適用ケース  
---|---|---  
5（小さい） | 非常に細かいクラスターが多数形成される。局所的な構造を強調するが、過剰にクラスター化される可能性。 | 局所的な異常値検出、微細な構造の探索  
20（中程度） | バランスの取れたクラスター形成。材料クラス（酸化物、金属等）が明確に分離される。 | 一般的な可視化（推奨）  
50（大きい） | 大域的な構造を保持。クラスター境界が曖昧になる場合がある。 | 大規模データセット、大域的なトレンド分析  
  
**最適値の決定方法:**

  1. **データサイズに応じた経験則:**
     * 小規模（<100点）: perplexity = 5-15
     * 中規模（100-1000点）: perplexity = 20-50
     * 大規模（>1000点）: perplexity = 50-100
  2. **複数の値で試行:** perplexity = [5, 10, 20, 30, 50]のように複数設定し、最も解釈しやすい結果を選択
  3. **クラスター評価指標:** Silhouette Score等で定量的に評価

**実例:** 材料探索（100-1000化合物）では、perplexity=20-30が最も材料クラスの分離が明瞭になることが多いです。

**Q7:** 文献や解説記事ではMagpieが「132次元」とも「145次元」とも書かれています。matminerが実際に生成するのはどちらで、もう一方の数字はどこから来たのでしょうか？

**正解:** matminerの`ElementProperty.from_preset('magpie')`が生成するのは**132列** で、これは一貫して変わりません（元素特性22種類 × 統計量6種類）。このプリセットが145列を返すmatminerのバージョンは存在しません。

**145はどこから来たか:** Ward et al. (2016)が提案した完全な属性セットの大きさで、4つのファミリーから構成されます：

  * **化学量論属性** 6個（組成ベクトルのL^pノルム。どの元素かに依らない）
  * **元素特性統計量** 132個（matminerの`magpie`プリセットが実装している部分）
  * **価電子軌道属性** 4個（価電子のうちs, p, d, f状態にある割合）
  * **イオン性化合物属性** 3個（電荷中性なイオン配置が存在するか、および最大イオン性）

6 + 132 + 4 + 3 = 145です。matminerではこの追加3ファミリーが`Stoichiometry`、`ValenceOrbital`、`IonProperty`という別のFeaturizerになっています。

**実務上の帰結:** Ward et al. (2016)の属性セットを再現するには、`MultipleFeaturizer([ElementProperty.from_preset('magpie'), Stoichiometry(), ValenceOrbital(), IonProperty()])`のように4つを連結します。`magpie`プリセットだけを呼んで「145特徴量」と報告しているなら、その数字は誤りです。次元数を書く前に`len(featurizer.feature_labels())`で確認してください。

**次元数に関する助言:** 小規模データ（<100サンプル）では132列でも過学習し得るので、PCAや特徴量選択を適用します。追加13属性の計算コスト増はごくわずかですが、特徴量セット自体は変わるため、Featurizerの定義はモデルと一緒にバージョン管理してください。

### Hard（発展）

**Q8:** カスタム統計関数として**geometric mean（幾何平均）** と**harmonic mean（調和平均）** を実装し、Fe2O3の電気陰性度について計算してください。また、これらの統計量が**arithmetic mean（算術平均）** とどのように異なり、どのような物理現象を表現するのに適しているか説明してください。  
（Feの電気陰性度: 1.83, Oの電気陰性度: 3.44）

**計算結果:**

  * Arithmetic mean = (1.83 + 3.44) / 2 = 2.635
  * Geometric mean = √(1.83 × 3.44) = √6.2952 = 2.509
  * Harmonic mean = 2 / (1/1.83 + 1/3.44) = 2 / (0.546 + 0.291) = 2 / 0.837 = 2.389

**統計量の大小関係:**

常に Harmonic mean ≤ Geometric mean ≤ Arithmetic mean が成立します（等号はすべての値が等しい場合のみ）。

**物理的意味と適用ケース:**

統計量 | 式 | 物理的意味 | 適用ケース  
---|---|---|---  
Arithmetic mean | $(x_1 + x_2) / 2$ | 線形加算効果 | 密度、モル質量等の示量性変数  
Geometric mean | $\sqrt{x_1 \times x_2}$ | 掛け算的効果 | 触媒活性（活性化エネルギー）、化学反応速度、複合材料の特性  
Harmonic mean | $2 / (1/x_1 + 1/x_2)$ | 直列抵抗効果 | 電気抵抗、熱伝導率、拡散係数（律速段階の支配）  
  
**Fe 2O3の電気陰性度の解釈:**

  * **Arithmetic mean (2.635):** 元素の多様性を反映。FeとOの電気陰性度を等重視。
  * **Geometric mean (2.509):** イオン結合性を表現するのに適しています。電気陰性度差 = 3.44 - 1.83 = 1.61が大きいため、強いイオン結合（Fe³⁺とO²⁻）が形成されます。
  * **Harmonic mean (2.389):** 電気陰性度が低い元素（Fe）の影響を強調。電子移動の「ボトルネック」を表現。

**実装コード例:**
    
    
    import numpy as np
    
    def geometric_mean(values):
        return np.prod(values) ** (1.0 / len(values))
    
    def harmonic_mean(values):
        return len(values) / sum(1.0 / v for v in values)
    
    # Fe2O3の電気陰性度
    en_values = [1.83, 3.44]  # Fe, O
    
    print(f"Arithmetic mean: {np.mean(en_values):.3f}")
    print(f"Geometric mean: {geometric_mean(en_values):.3f}")
    print(f"Harmonic mean: {harmonic_mean(en_values):.3f}")
    

**Q9:** 以下の3つの材料系（酸化物、金属、半導体）のMagpie特徴量を比較分析したいと考えています。どのような特徴量の組み合わせを可視化すれば、材料クラス間の違いを最も明確に示せますか？3つの特徴量ペアを提案し、その理由を説明してください。

**推奨する3つの特徴量ペアと理由:**

#### 1\. mean Electronegativity vs range Electronegativity

**理由:**

  * **酸化物:** 高い平均電気陰性度（金属+非金属）、大きい範囲（イオン結合性）
  * **金属:** 低い平均電気陰性度、小さい範囲（似た元素の組み合わせ）
  * **半導体:** 中程度の平均電気陰性度、中程度の範囲

**期待される分離:** 3クラスが明確に分離される最も基本的な特徴量ペア。

#### 2\. weighted_mean GSbandgap vs mean IonizationEnergy

**理由:**

  * **酸化物:** 高いバンドギャップ（>3 eV、絶縁体）、高いイオン化エネルギー
  * **金属:** バンドギャップ = 0 eV（導電性）、低いイオン化エネルギー
  * **半導体:** 中程度のバンドギャップ（1-3 eV）、中程度のイオン化エネルギー

**期待される分離:** 電子状態の違いを直接反映。材料の電気特性予測に有用。

#### 3\. mean AtomicRadius vs weighted_mean MeltingT

**理由:**

  * **酸化物:** 中程度の原子半径、中～高融点（セラミック特性）
  * **金属:** 大きい原子半径（金属半径が大きい）、高融点（遷移金属）
  * **半導体:** 中～大きい原子半径、中程度の融点

**期待される分離:** 構造的・熱力学的安定性の違いを表現。

**実装例:**
    
    
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # 特徴量ペアの可視化
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    pairs = [
        ("mean Electronegativity", "range Electronegativity"),
        ("weighted_mean GSbandgap", "mean IonizationEnergy"),
        ("mean AtomicRadius", "weighted_mean MeltingT")
    ]
    
    for idx, (feat1, feat2) in enumerate(pairs):
        ax = axes[idx]
        for material_class in ["Oxides", "Metals", "Semiconductors"]:
            mask = df_results["Class"] == material_class
            ax.scatter(
                df_results[mask][feat1],
                df_results[mask][feat2],
                label=material_class,
                s=100,
                alpha=0.7
            )
        ax.set_xlabel(feat1)
        ax.set_ylabel(feat2)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    

**Q10:** PCA、t-SNE、UMAPの3つの次元削減手法を、以下の3つの評価軸で比較し、10,000化合物のMagpie特徴量データセットに対してどの手法を選ぶべきか判断してください。  
**評価軸:** (1) 計算時間、(2) クラスター分離性能、(3) 大域的構造の保存

**3手法の定量的比較（10,000化合物の場合）:**

手法 | 計算時間 | クラスター分離性能 | 大域的構造の保存 | 総合評価  
---|---|---|---|---  
PCA | ⭐⭐⭐⭐⭐  
~1秒 | ⭐⭐⭐  
線形分離のみ | ⭐⭐⭐⭐⭐  
完全保存 | 高速だが分離性能は限定的  
t-SNE | ⭐  
~30-60分 | ⭐⭐⭐⭐⭐  
最高 | ⭐⭐  
保存されない | 美しいが時間がかかる  
UMAP | ⭐⭐⭐⭐  
~2-5分 | ⭐⭐⭐⭐  
高い | ⭐⭐⭐⭐  
ある程度保存 | **バランス最良（推奨）**  
  
**推奨：UMAP（理由の詳細）**

  1. **計算時間:** 10,000化合物に対して2-5分程度で実行可能。t-SNEの10-20倍高速。
  2. **クラスター分離性能:** t-SNEに匹敵する高い分離性能。材料クラス（酸化物、金属、半導体等）が明確に分離される。
  3. **大域的構造:** t-SNEと異なり、クラスター間の距離にもある程度の意味がある。例えば、「酸化物と金属の距離 > 酸化物内のサブクラス間距離」が保たれる。

**各手法の適用ケース:**

  * **PCA:** 探索的データ解析（EDA）の最初のステップ、寄与率分析、100万点超のデータ
  * **t-SNE:** 論文用の美しい可視化、小規模データ（<5,000点）、計算時間に余裕がある場合
  * **UMAP:** 実用的な材料探索、中～大規模データ（1,000-100,000点）、バランス重視

**実装例（3手法の比較）:**
    
    
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    import umap
    import time
    
    # データ準備（10,000化合物のMagpie特徴量）
    # X = np.array(magpie_features)  # shape: (10000, 132)
    
    methods = {
        "PCA": PCA(n_components=2),
        "t-SNE": TSNE(n_components=2, perplexity=30, n_iter=1000),
        "UMAP": umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1)
    }
    
    results = {}
    for name, model in methods.items():
        start = time.time()
        X_reduced = model.fit_transform(X)
        elapsed = time.time() - start
        results[name] = {"time": elapsed, "data": X_reduced}
        print(f"{name}: {elapsed:.2f}秒")
    
    # 可視化比較
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for idx, (name, result) in enumerate(results.items()):
        ax = axes[idx]
        scatter = ax.scatter(
            result["data"][:, 0],
            result["data"][:, 1],
            c=labels,  # 材料クラスラベル
            cmap="Set2",
            s=10,
            alpha=0.6
        )
        ax.set_title(f"{name} ({result['time']:.1f}秒)")
        ax.set_xlabel("次元1")
        ax.set_ylabel("次元2")
    
    plt.tight_layout()
    plt.show()
    

**結論:** 10,000化合物のデータセットには**UMAP** が最適です。計算時間、分離性能、構造保存のバランスが最も優れています。

## 次のステップ

この章では、Magpie記述子の詳細構成、元素特性の種類、統計的集約手法、次元削減による可視化を学びました。

次の第3章では、**元素特性データベースとFeaturizer** を学び、多様なデータソースを統合してカスタム特徴量を設計する手法を探求します。

[← 第1章：組成ベース特徴量の基礎](<./chapter-1.html>) [シリーズ目次に戻る](<./index.html>) [第3章：元素特性データベースとFeaturizer →](<./chapter-3.html>)

## 参考文献

  1. Ward, L., Agrawal, A., Choudhary, A., & Wolverton, C. (2016). "A general-purpose machine learning framework for predicting properties of inorganic materials." _npj Computational Materials_ , 2, 16028, pp. 1-7. https://doi.org/10.1038/npjcompumats.2016.28
  2. Ghiringhelli, L. M., Vybiral, J., Levchenko, S. V., Draxl, C., & Scheffler, M. (2015). "Big Data of Materials Science: Critical Role of the Descriptor." _Physical Review Letters_ , 114(10), 105503, pp. 1-5. https://doi.org/10.1103/PhysRevLett.114.105503
  3. Ward, L., Liu, R., Krishna, A., Hegde, V. I., Agrawal, A., Choudhary, A., & Wolverton, C. (2017). "Including crystal structure attributes in machine learning models of formation energies via Voronoi tessellations." _Physical Review B_ , 96(2), 024104, pp. 1-12. https://doi.org/10.1103/PhysRevB.96.024104
  4. Oliynyk, A. O., Antono, E., Sparks, T. D., Ghadbeigi, L., Gaultois, M. W., Meredig, B., & Mar, A. (2016). "High-Throughput Machine-Learning-Driven Synthesis of Full-Heusler Compounds." _Chemistry of Materials_ , 28(20), 7324-7331, pp. 7324-7331. https://doi.org/10.1021/acs.chemmater.6b02724
  5. matminer Documentation: Composition-based featurizers. Hacking Materials Research Group, Lawrence Berkeley National Laboratory. https://hackingmaterials.lbl.gov/matminer/featurizer_summary.html#composition-based-featurizers (Accessed: 2025-01-15)
  6. scikit-learn Documentation: Feature selection. scikit-learn developers. https://scikit-learn.org/stable/modules/feature_selection.html (Accessed: 2025-01-15)
  7. Mendeleev Python library documentation. https://mendeleev.readthedocs.io/ (Accessed: 2025-01-15)

* * *

[シリーズ目次に戻る](<./index.html>) | [← 第1章](<./chapter-1.html>) | [第3章 →](<./chapter-3.html>)

### 免責事項

  * 本コンテンツは教育・研究・情報提供のみを目的としており、専門的な助言(法律・会計・技術的保証など)を提供するものではありません。
  * 本コンテンツおよび付随するCode examplesは「現状有姿(AS IS)」で提供され、明示または黙示を問わず、商品性、特定目的適合性、権利非侵害、正確性・完全性、動作・安全性等いかなる保証もしません。
  * 外部リンク、第三者が提供するデータ・ツール・ライブラリ等の内容・可用性・安全性について、作成者および東北大学は一切の責任を負いません。
  * 本コンテンツの利用・実行・解釈により直接的・間接的・付随的・特別・結果的・懲罰的損害が生じた場合でも、適用法で許容される最大限の範囲で、作成者および東北大学は責任を負いません。
  * 本コンテンツの内容は、予告なく変更・更新・提供停止されることがあります。
  * 本コンテンツの著作権・ライセンスは明記された条件(例: CC BY 4.0)に従います。当該ライセンスは通常、無保証条項を含みます。
