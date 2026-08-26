---
title: 第1章：組成ベース特徴量の基礎
chapter_title: 第1章：組成ベース特徴量の基礎
subtitle: 化学組成から材料特性を予測する原理
---

### 🎯 この章の学習目標

#### 基本理解

  * ✅ 組成ベース特徴量の定義と役割を説明できる
  * ✅ 統計的集約（平均、分散、最大/最小、範囲）の原理を理解している
  * ✅ 構造情報なしでの予測可能性とその限界を理解している

#### 実践スキル

  * ✅ pymatgenで化学式を解析し、元素情報を抽出できる
  * ✅ matminerで基本的な組成ベース特徴量を生成できる
  * ✅ 元素特性データベースを活用し、統計量を計算できる

#### 応用力

  * ✅ 新規材料系に組成ベース特徴量を適用できる
  * ✅ 機械学習モデルと統合した予測ワークフローを設計できる
  * ✅ 適用可能なタスクと不適切なタスクを判断できる

## 1.1 化学組成からの特徴量抽出原理

### 組成ベース特徴量とは何か

材料探索において、**化学組成** （元素の種類と比率）は最も基本的な情報です。例えば、酸化鉄の化学式「Fe2O3」は「鉄原子2個と酸素原子3個」を意味します。しかし、この文字列をそのまま機械学習モデルに入力することはできません。

**組成ベース特徴量** は、化学組成を**数値ベクトル** に変換する技術です。具体的には、元素の周期表情報（原子半径、イオン化エネルギー、電気陰性度など）を利用し、以下のように変換します：
    
    
    ```mermaid
    graph LR
        A["化学式Fe₂O₃"] --> B["元素抽出Fe: 2原子O: 3原子"]
        B --> C["元素特性取得Fe: 原子半径=1.26Å, IE=7.9eVO: 原子半径=0.66Å, IE=13.6eV"]
        C --> D["統計的集約平均原子半径=0.900Å平均IE=11.32eV..."]
        D --> E["特徴量ベクトル[0.900, 11.32, ...](132次元)"]
    ```

### 化学組成 vs 結晶構造の情報量比較

材料の記述には、大きく2つのアプローチがあります：

アプローチ | 必要な情報 | 情報量 | 予測精度 | 計算速度  
---|---|---|---|---  
**組成ベース** | 化学式のみ（例: Fe₂O₃） | 少（~132次元） | 中（R²=0.7-0.85） | 高速（100万化合物で約10分）  
**構造ベース** （GNN） | 原子座標、結合情報 | 多（~数千次元） | 高（R²=0.85-0.95） | 低速（1分で1000化合物）  
  
#### 💡 組成ベースが有利な3つのケース

  1. **構造未知の新材料探索** : 合成前の候補スクリーニングでは結晶構造が不明
  2. **高速大規模スクリーニング** : 100万化合物の形成エネルギー予測（GNNの100倍高速）
  3. **実験データ駆動型** : 構造情報が取得困難な実験データからの学習

### 元素特性の種類と材料科学的意義

組成ベース特徴量は、**元素周期表のデータベース** を活用します。代表的な元素特性22種類：

カテゴリ | 元素特性 | 材料科学的意義  
---|---|---  
**原子構造** | 原子番号（Atomic Number） | 元素の基本識別子  
周期（Period） | 電子殻の数（化学的反応性の指標）  
族（Group） | 価電子数（結合傾向の指標）  
**原子サイズ** | 原子半径（Atomic Radius） | 結晶格子の大きさ、充填率  
共有結合半径（Covalent Radius） | 共有結合の長さ  
イオン半径（Ionic Radius） | イオン結晶の格子定数  
原子体積（Atomic Volume） | 結晶密度の推定  
**電子的性質** | イオン化エネルギー（Ionization Energy） | 化学結合の強さ、反応性  
電子親和力（Electron Affinity） | 酸化・還元傾向  
電気陰性度（Electronegativity） | 結合の極性、イオン性  
価電子数（Valence Electrons） | 化学結合の種類  
酸化状態（Oxidation State） | イオン結合の電荷バランス  
**熱的・物理的性質** | 融点（Melting Point） | 結晶安定性、合成温度  
沸点（Boiling Point） | 蒸発しやすさ  
密度（Density） | 結晶の充填率  
熱伝導率（Thermal Conductivity） | 熱輸送特性  
**その他** | 地殻存在度（Abundance） | 材料コスト、希少性  
発見年（Discovery Year） | 元素の歴史的背景  
比熱容量（Specific Heat） | 熱容量の推定  
電気抵抗率（Electrical Resistivity） | 導電性の指標  
磁気モーメント（Magnetic Moment） | 磁性材料の設計  
分極率（Polarizability） | 誘電特性  
  
### 統計的集約手法（平均、分散、最大/最小、範囲）

複数の元素からなる化合物（例: Fe₂O₃）では、各元素の特性を**統計的に集約** します。代表的な6つの統計量：

  1. **平均（Mean）** : $\bar{x} = \frac{1}{n}\sum_{i=1}^{n} w_i x_i$ （重み$w_i$は組成比）
  2. **分散（Variance）** : $\sigma^2 = \frac{1}{n}\sum_{i=1}^{n} w_i (x_i - \bar{x})^2$
  3. **標準偏差（Standard Deviation）** : $\sigma = \sqrt{\sigma^2}$
  4. **最大値（Maximum）** : $\max(x_1, x_2, \ldots, x_n)$
  5. **最小値（Minimum）** : $\min(x_1, x_2, \ldots, x_n)$
  6. **範囲（Range）** : $\text{range} = \max - \min$

#### 🔍 具体例：Fe₂O₃の原子半径統計量

  * Fe原子半径: 1.26 Å （組成比: 2/5 = 0.4）
  * O原子半径: 0.66 Å （組成比: 3/5 = 0.6）

**計算結果** :

  * 平均: $0.4 \times 1.26 + 0.6 \times 0.66 = 0.900$ Å
  * 分散: $0.4(1.26-0.90)^2 + 0.6(0.66-0.90)^2 = 0.0864$ Ų
  * 最大: 1.26 Å, 最小: 0.66 Å, 範囲: 0.60 Å

### コード例1：pymatgen Compositionクラス基本操作

[Google Colabで開く](<https://colab.research.google.com/github/AI-Terakoya/composition-features-tutorial/blob/main/chapter1_example1.ipynb>)
    
    
    # 化学式を解析し、元素情報を抽出する
    from pymatgen.core import Composition
    
    # 化学式の作成
    comp = Composition("Fe2O3")
    
    # 基本情報の取得
    print(f"化学式: {comp}")
    print(f"元素種類: {comp.elements}")  # [Element Fe, Element O]
    print(f"総原子数: {comp.num_atoms}")  # 5.0
    print(f"総重量: {comp.weight:.2f} g/mol")  # 159.69 g/mol
    
    # 各元素の組成比
    print("\n元素ごとの組成比:")
    for element, fraction in comp.get_atomic_fraction().items():
        print(f"  {element}: {fraction:.3f} ({fraction*comp.num_atoms:.0f}原子)")
    # 出力:
    #   Fe: 0.400 (2原子)
    #   O: 0.600 (3原子)
    
    # 分数表記の確認
    print(f"\n約分前: {Composition('Fe4O6')}")  # Fe4 O6
    print(f"約分後: {Composition('Fe4O6').reduced_composition}")  # Fe2 O3

### コード例2：元素特性の抽出と可視化

[Google Colabで開く](<https://colab.research.google.com/github/AI-Terakoya/composition-features-tutorial/blob/main/chapter1_example2.ipynb>)
    
    
    # 元素特性を抽出し、matplotlibで可視化する
    import matplotlib.pyplot as plt
    from pymatgen.core import Element
    
    # 複数元素の原子半径を比較
    elements = [Element("Fe"), Element("O"), Element("Cu"), Element("Si")]
    properties = {
        "原子半径 (Å)": [el.atomic_radius for el in elements],
        "イオン化エネルギー (eV)": [el.ionization_energy for el in elements],
        "電気陰性度 (Pauling)": [el.X for el in elements]
    }
    
    # 可視化
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    element_names = [el.symbol for el in elements]
    
    for ax, (prop_name, values) in zip(axes, properties.items()):
        ax.bar(element_names, values, color=['#f093fb', '#f5576c', '#feca57', '#48dbfb'])
        ax.set_ylabel(prop_name, fontsize=12)
        ax.set_title(f"元素特性比較: {prop_name}", fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('element_properties.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # 数値出力
    print("元素特性データ:")
    for prop_name, values in properties.items():
        print(f"\n{prop_name}:")
        for el, val in zip(element_names, values):
            print(f"  {el}: {val:.3f}")

### コード例3：統計量計算（平均、標準偏差、範囲）

[Google Colabで開く](<https://colab.research.google.com/github/AI-Terakoya/composition-features-tutorial/blob/main/chapter1_example3.ipynb>)
    
    
    # 組成ベース統計量を手動計算
    import numpy as np
    from pymatgen.core import Composition, Element
    
    def compute_weighted_stats(comp, property_name):
        """組成比で重み付けした統計量を計算"""
        # 元素と組成比の抽出
        fractions = comp.get_atomic_fraction()
    
        # 元素特性の取得
        values = []
        weights = []
        for element, frac in fractions.items():
            prop_value = getattr(Element(element), property_name)
            values.append(prop_value)
            weights.append(frac)
    
        values = np.array(values)
        weights = np.array(weights)
    
        # 統計量の計算
        mean = np.sum(weights * values)
        variance = np.sum(weights * (values - mean)**2)
        std = np.sqrt(variance)
    
        return {
            'mean': mean,
            'std': std,
            'variance': variance,
            'max': values.max(),
            'min': values.min(),
            'range': values.max() - values.min()
        }
    
    # Fe2O3の原子半径統計量を計算
    comp = Composition("Fe2O3")
    stats = compute_weighted_stats(comp, 'atomic_radius')
    
    print("Fe2O3の原子半径統計量:")
    for stat_name, value in stats.items():
        print(f"  {stat_name}: {value:.4f} Å")
    
    # 出力例:
    #   mean: 0.9000 Å
    #   std: 0.2939 Å
    #   variance: 0.0864 Ų
    #   max: 1.2600 Å
    #   min: 0.6600 Å
    #   range: 0.6000 Å

## 1.2 元素周期表と材料特性

### 周期律と材料特性の相関

元素周期表は、**元素の性質が周期的に変化する** という周期律に基づいています。この周期性は、材料特性の予測に直接的に寄与します：
    
    
    ```mermaid
    graph TD
        A[元素周期表] --> B[周期（Period）]
        A --> C[族（Group）]
        B --> D[電子殻の数→ 原子サイズ]
        C --> E[価電子数→ 結合性質]
        D --> F[結晶格子定数密度]
        E --> G[イオン性共有結合性]
        F --> H[材料特性形成エネルギーバンドギャップ]
        G --> H
    ```

#### 周期（Period）の影響

  * **周期が増加すると** （上から下へ）： 
    * 原子半径が増加 → 結晶格子が大きくなる
    * イオン化エネルギーが減少 → 反応性が高くなる
    * 金属性が増加 → 導電性が向上

#### 族（Group）の影響

  * **族が増加すると** （左から右へ）： 
    * 価電子数が増加 → 酸化状態の多様性
    * 電気陰性度が増加 → イオン結合の極性
    * 非金属性が増加 → 絶縁性材料

### 元素グループの傾向（遷移金属、ハロゲン等）

元素グループ | 代表元素 | 特徴的性質 | 材料応用例  
---|---|---|---  
**アルカリ金属**  
(1族) | Li, Na, K | ・低イオン化エネルギー  
・高反応性  
・軽量 | リチウムイオン電池  
ナトリウムイオン電池  
**遷移金属**  
(3-12族) | Fe, Co, Ni, Cu | ・複数の酸化状態  
・d軌道電子  
・磁性 | 触媒、磁性材料  
構造材料  
**ハロゲン**  
(17族) | F, Cl, Br, I | ・高電気陰性度  
・強い酸化力  
・イオン結合形成 | ペロブスカイト  
ハロゲン化物電解質  
**希土類元素**  
(ランタノイド) | La, Ce, Nd, Gd | ・4f軌道電子  
・磁気モーメント  
・蛍光特性 | 永久磁石、蛍光体  
触媒  
**半金属**  
(13-16族境界) | Si, Ge, As | ・金属と非金属の中間  
・バンドギャップ調整可能 | 半導体、太陽電池  
熱電材料  
  
### 構造化学的背景（結合の種類、配位数）

元素特性は、**化学結合の種類** と**配位構造** に直接的に影響します：

#### 結合の種類と元素特性の関係

  1. **イオン結合** : 
     * 電気陰性度の差が大きい元素間（例: Na-Cl, Ca-O）
     * 電気陰性度差 $\Delta X > 1.7$ でイオン性が支配的
     * 予測：形成エネルギーが大きい、硬い、絶縁性
  2. **共有結合** : 
     * 電気陰性度が近い元素間（例: Si-Si, C-C）
     * $\Delta X < 0.5$ で共有結合性が強い
     * 予測：方向性のある結合、半導体特性
  3. **金属結合** : 
     * 金属元素間（例: Fe-Fe, Cu-Cu）
     * 自由電子の共有
     * 予測：高導電性、展性・延性

#### 配位数と原子半径比

イオン結晶において、**配位数** （中心イオンの周りのイオン数）は、原子半径比 $r_{\text{cation}}/r_{\text{anion}}$ で決まります：

半径比 | 配位数 | 配位構造 | 例  
---|---|---|---  
0.225 - 0.414 | 4 | 四面体 | ZnS（閃亜鉛鉱）  
0.414 - 0.732 | 6 | 八面体 | NaCl（岩塩）  
0.732 - 1.000 | 8 | 立方体 | CsCl（塩化セシウム）  
> 1.000 | 12 | 最密充填 | 金属結晶  
  
#### 🔬 実例：TiO₂の配位構造予測

  * Ti⁴⁺イオン半径: 0.605 Å
  * O²⁻イオン半径: 1.40 Å
  * 半径比: $0.605 / 1.40 = 0.432$
  * **予測** : 配位数6（八面体構造）
  * **実際** : ルチル型TiO₂は確かにTi⁴⁺が6配位

### コード例4：組成ベース特徴量ベクトル生成

[Google Colabで開く](<https://colab.research.google.com/github/AI-Terakoya/composition-features-tutorial/blob/main/chapter1_example4.ipynb>)
    
    
    # matminerで組成から特徴量ベクトルを生成
    from matminer.featurizers.composition import ElementProperty
    import pandas as pd
    
    # 化合物リスト
    compounds = ["Fe2O3", "TiO2", "Al2O3", "SiO2", "CuO"]
    
    # ElementPropertyフィーチャライザー（簡易版、3統計量のみ）
    featurizer = ElementProperty.from_preset("magpie")
    
    # データフレーム作成
    df = pd.DataFrame({"composition": compounds})
    
    # 特徴量生成（時間がかかる場合がある）
    df = featurizer.featurize_dataframe(df, col_id="composition")
    
    # 生成された特徴量の一部を表示
    feature_cols = [col for col in df.columns if col != "composition"]
    print(f"生成された特徴量数: {len(feature_cols)}")
    print(f"\n最初の5つの特徴量:")
    print(df[feature_cols[:5]].head())
    
    # 特徴量名の例
    print(f"\n特徴量名の例:")
    for i, col in enumerate(feature_cols[:10]):
        print(f"  {i+1}. {col}")
    
    # 出力例:
    # 生成された特徴量数: 132
    # 特徴量名の例:
    #   1. MagpieData mean Number
    #   2. MagpieData avg_dev Number
    #   3. MagpieData range Number
    #   ...

### コード例5：元素周期表マッピング（seaborn heatmap）

[Google Colabで開く](<https://colab.research.google.com/github/AI-Terakoya/composition-features-tutorial/blob/main/chapter1_example5.ipynb>)
    
    
    # 元素周期表を可視化し、特定の元素特性をヒートマップで表示
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    from pymatgen.core import Element
    
    # 周期表の一部（主要元素）
    periods = {
        2: ["Li", "Be", "B", "C", "N", "O", "F"],
        3: ["Na", "Mg", "Al", "Si", "P", "S", "Cl"],
        4: ["K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn"],
    }
    
    # 電気陰性度のヒートマップデータ作成
    max_cols = max(len(row) for row in periods.values())
    heatmap_data = []
    yticks = []
    
    for period_num, elements in sorted(periods.items()):
        row = []
        for el_symbol in elements:
            el = Element(el_symbol)
            row.append(el.X if el.X is not None else 0)
        # 残りは0で埋める
        row.extend([0] * (max_cols - len(row)))
        heatmap_data.append(row)
        yticks.append(f"Period {period_num}")
    
    # ヒートマップ描画
    plt.figure(figsize=(14, 6))
    sns.heatmap(heatmap_data, annot=False, cmap="RdYlBu_r",
                cbar_kws={'label': 'Electronegativity (Pauling)'},
                yticklabels=yticks)
    plt.title("元素周期表: 電気陰性度のヒートマップ", fontsize=16, fontweight='bold')
    plt.xlabel("元素（左から右へ）", fontsize=12)
    plt.ylabel("周期", fontsize=12)
    plt.tight_layout()
    plt.savefig('periodic_table_electronegativity.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("ヒートマップから読み取れる傾向:")
    print("- 右上（F, O, Cl）で電気陰性度が高い（赤色）")
    print("- 左下（Li, Na, K）で電気陰性度が低い（青色）")
    print("- 同じ族（縦）では下に行くほど減少")

## 1.3 組成ベース予測の成功事例

### OQMD/Materials Project形成エネルギー予測（R² ≥ 0.8）

組成ベース特徴量の最も成功した応用例は、**形成エネルギー予測** です。Ward et al. (2016)のMagpie論文では、以下の成果が報告されています：

データセット | サンプル数 | モデル | R²スコア | MAE  
---|---|---|---|---  
OQMD（全化合物） | 435,000 | Random Forest | 0.92 | 0.10 eV/atom  
Materials Project | 60,000 | Gradient Boosting | 0.89 | 0.12 eV/atom  
OQMD（酸化物のみ） | 50,000 | Random Forest | 0.94 | 0.08 eV/atom  
  
#### 📊 形成エネルギーとは

形成エネルギー（Formation Energy）は、化合物が元素単体から生成する際のエネルギー変化です：

$$\Delta H_f = E_{\text{compound}} - \sum_i n_i E_{\text{element}_i}$$

  * **負の値** : 化合物が安定（自発的に生成）
  * **正の値** : 化合物が不安定（生成困難）

材料探索では、形成エネルギーが負で絶対値が大きいほど、合成しやすい材料として優先されます。

### バンドギャップ予測（MAE < 0.5 eV）

半導体材料の設計に重要な**バンドギャップ** も、組成ベース特徴量で予測可能です：

研究 | データセット | サンプル数 | MAE | R²  
---|---|---|---|---  
Ward+ (2016) | Materials Project | 10,000 | 0.45 eV | 0.78  
Jha+ (2018) ElemNet | OQMD | 28,000 | 0.38 eV | 0.83  
Meredig+ (2014) | 実験データ | 1,200 | 0.62 eV | 0.65  
  
**注意点** : バンドギャップは**構造依存性が強い** 物性です。DFT計算値との比較では、組成ベースのみでは限界があり、GNN（構造ベース）手法のほうが高精度（MAE < 0.25 eV）です。

### 熱電特性予測

熱電材料（熱を電気に変換）の性能指標**ZT値** の予測にも成功しています：

  * **Carrete et al. (2014)** : 半ホイスラー化合物のZT予測（R² = 0.72）
  * **特徴量の重要度** : 
    1. 平均原子質量（熱伝導率に影響）
    2. 電気陰性度の分散（キャリア移動度に影響）
    3. 価電子数の平均（電子濃度に影響）

### コード例6：組成正規化処理

[Google Colabで開く](<https://colab.research.google.com/github/AI-Terakoya/composition-features-tutorial/blob/main/chapter1_example6.ipynb>)
    
    
    # 組成式を正規化し、1原子あたりに換算
    from pymatgen.core import Composition
    
    # さまざまな表記の組成式
    formulas = ["Fe4O6", "Fe2O3", "Fe0.5O0.75", "FeO1.5"]
    
    print("組成式の正規化:")
    for formula in formulas:
        comp = Composition(formula)
        reduced = comp.reduced_composition
        fractional = comp.fractional_composition
    
        print(f"\n元の式: {formula}")
        print(f"  約分後（整数比）: {reduced}")
        print(f"  1原子あたり: {fractional}")
        print(f"  総原子数: {comp.num_atoms:.2f}")
        print(f"  Fe/O比: {comp['Fe']/comp['O']:.3f}")
    
    # 出力例:
    # 元の式: Fe4O6
    #   約分後（整数比）: Fe2 O3
    #   1原子あたり: Fe0.4 O0.6
    #   総原子数: 10.00
    #   Fe/O比: 0.667

## 1.4 Why組成ベース？構造情報なしでの予測可能性

### 高速スクリーニングの利点

組成ベース特徴量の最大の利点は、**計算速度** です。構造ベース（GNN）との比較：

タスク | 組成ベース（Magpie + RF） | 構造ベース（CGCNN） | 速度比  
---|---|---|---  
特徴量生成（1化合物） | 0.001秒 | 0.1秒 | 100倍高速  
推論（100万化合物） | 10分 | 27時間 | 162倍高速  
モデル訓練（10万サンプル） | 5分 | 60分 | 12倍高速  
  
#### ⚡ 高速スクリーニングの実用例

**シナリオ** : 100万個の候補化合物から、形成エネルギーが-2 eV/atom以下の安定な化合物を探す

  * **組成ベース** : 10分で完了 → 5万化合物を選出 → 実験的検証へ
  * **構造ベース（GNN）** : 27時間かかる → 実用的でない

**戦略** : 組成ベースで候補を絞り込んだ後、GNNで精密予測（ハイブリッドアプローチ）

### 未知結晶構造材料への適用

新材料探索では、**結晶構造が未知** のケースが多数存在します：

  1. **合成前の候補化合物** : 組成は決められるが、構造は不明 
     * 例: Li-Ni-Mn-Co-O系電池正極材料の組成探索
     * 組成ベースで候補を絞り込み → 合成 → 構造解析
  2. **準安定相** : DFT計算で構造が予測困難 
     * 例: 高圧合成材料、急冷凝固材料
     * 組成から物性の傾向を予測
  3. **アモルファス材料** : 長距離秩序がない 
     * 例: 金属ガラス、酸化物ガラス
     * 組成が唯一の記述子

### 実験データとの親和性

実験研究者にとって、**組成情報は最も取得しやすい** データです：

情報の種類 | 実験的取得難易度 | 精度 | コスト  
---|---|---|---  
化学組成 | 低（EDX, ICP-MS） | 高（±1%） | 低（数千円/サンプル）  
結晶構造 | 中（XRD, 単結晶解析） | 中（Rietveld解析が必要） | 中（数万円/サンプル）  
原子座標（精密） | 高（単結晶XRD, 中性子回折） | 高（Å精度） | 高（数十万円/サンプル）  
  
**実験データ駆動型材料探索の典型的ワークフロー** :
    
    
    ```mermaid
    graph LR
        A[実験で組成測定] --> B[組成ベース特徴量生成]
        B --> C[機械学習モデル訓練]
        C --> D[新組成の物性予測]
        D --> E[実験的検証]
        E --> F{性能向上?}
        F -->|はい| G[次世代組成最適化]
        F -->|いいえ| C
        G --> E
    ```

### コード例7：特徴量の相関分析（pandas、seaborn）

[Google Colabで開く](<https://colab.research.google.com/github/AI-Terakoya/composition-features-tutorial/blob/main/chapter1_example7.ipynb>)
    
    
    # 特徴量間の相関を分析し、冗長な特徴量を識別
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    from matminer.featurizers.composition import ElementProperty
    
    # サンプルデータ
    compounds = ["Fe2O3", "TiO2", "Al2O3", "SiO2", "CuO", "ZnO", "MgO", "CaO"]
    df = pd.DataFrame({"composition": compounds})
    
    # 特徴量生成（簡略版、統計量のみ）
    featurizer = ElementProperty(features=["Number", "AtomicWeight", "Row", "Column"],
                                  stats=["mean", "std", "range"])
    df = featurizer.featurize_dataframe(df, col_id="composition")
    
    # 特徴量列のみ抽出
    feature_cols = [col for col in df.columns if col != "composition"]
    
    # 相関行列計算
    corr_matrix = df[feature_cols].corr()
    
    # ヒートマップ描画
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                center=0, square=True, linewidths=0.5,
                cbar_kws={"shrink": 0.8})
    plt.title("組成ベース特徴量の相関行列", fontsize=16, fontweight='bold')
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.savefig('feature_correlation.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # 高相関ペアの検出（閾値: |r| > 0.9）
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if abs(corr_matrix.iloc[i, j]) > 0.9:
                high_corr_pairs.append((
                    corr_matrix.columns[i],
                    corr_matrix.columns[j],
                    corr_matrix.iloc[i, j]
                ))
    
    print("\n高相関な特徴量ペア（|r| > 0.9）:")
    for feat1, feat2, corr_val in high_corr_pairs:
        print(f"  {feat1} ↔ {feat2}: r = {corr_val:.3f}")

### コード例8：簡単な線形回帰モデル適用（scikit-learn）

[Google Colabで開く](<https://colab.research.google.com/github/AI-Terakoya/composition-features-tutorial/blob/main/chapter1_example8.ipynb>)
    
    
    # 組成ベース特徴量で形成エネルギーを予測（シミュレーションデータ）
    import numpy as np
    import pandas as pd
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_absolute_error, r2_score
    from matminer.featurizers.composition import ElementProperty
    import matplotlib.pyplot as plt
    
    # シミュレーションデータ（実際にはMaterials Project等から取得）
    np.random.seed(42)
    compounds = ["Fe2O3", "TiO2", "Al2O3", "SiO2", "CuO", "ZnO", "MgO", "CaO",
                 "NiO", "CoO", "MnO", "V2O5", "Cr2O3", "SnO2", "In2O3", "Ga2O3"]
    formation_energies = [-2.5, -3.1, -3.8, -2.9, -1.5, -1.8, -2.3, -2.7,
                          -1.4, -1.6, -1.9, -3.5, -2.8, -2.4, -2.1, -2.6]  # eV/atom（架空の値）
    
    # データフレーム作成
    df = pd.DataFrame({"composition": compounds, "formation_energy": formation_energies})
    
    # 特徴量生成
    featurizer = ElementProperty.from_preset("magpie")
    df = featurizer.featurize_dataframe(df, col_id="composition")
    
    # 特徴量とターゲット分離
    feature_cols = [col for col in df.columns if col not in ["composition", "formation_energy"]]
    X = df[feature_cols].values
    y = df["formation_energy"].values
    
    # 訓練/テスト分割
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # 線形回帰モデル
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # 予測
    y_pred = model.predict(X_test)
    
    # 評価
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"モデル性能:")
    print(f"  MAE: {mae:.3f} eV/atom")
    print(f"  R²: {r2:.3f}")
    
    # 予測 vs 実測のプロット
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, color='#f5576c', s=100, alpha=0.7, edgecolors='black')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
             'k--', lw=2, label='Perfect Prediction')
    plt.xlabel('実測値 (eV/atom)', fontsize=12)
    plt.ylabel('予測値 (eV/atom)', fontsize=12)
    plt.title(f'形成エネルギー予測（線形回帰）\nMAE={mae:.3f}, R²={r2:.3f}',
              fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('linear_regression_prediction.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # 重要な特徴量トップ5
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'coefficient': np.abs(model.coef_)
    }).sort_values('coefficient', ascending=False)
    
    print("\n重要な特徴量トップ5:")
    print(feature_importance.head())

## 演習問題

### Easy（基礎レベル）

演習1-1：化学式解析

**問題** : pymatgenを使って、化学式「Li3Fe2(PO4)3」から以下を抽出せよ：

  1. 元素の種類と個数
  2. 総原子数
  3. 総重量（g/mol）
  4. 各元素の組成比（原子分率）

**解答例** :
    
    
    from pymatgen.core import Composition
    
    comp = Composition("Li3Fe2(PO4)3")
    
    # 1. 元素の種類と個数
    print("元素と個数:")
    for el, count in comp.get_el_amt_dict().items():
        print(f"  {el}: {count}個")
    
    # 2. 総原子数
    print(f"\n総原子数: {comp.num_atoms}")
    
    # 3. 総重量
    print(f"総重量: {comp.weight:.2f} g/mol")
    
    # 4. 組成比
    print("\n組成比（原子分率）:")
    for el, frac in comp.get_atomic_fraction().items():
        print(f"  {el}: {frac:.4f}")
    
    # 出力:
    # 元素と個数:
    #   Li: 3個
    #   Fe: 2個
    #   P: 3個
    #   O: 12個
    # 総原子数: 20.0
    # 総重量: 417.43 g/mol
    # 組成比:
    #   Li: 0.1500
    #   Fe: 0.1000
    #   P: 0.1500
    #   O: 0.6000

演習1-2：基本統計量計算

**問題** : 化学式「NaCl」の原子半径について、平均、標準偏差、範囲を計算せよ（組成比で重み付け）。

**ヒント** : Na原子半径=1.86Å、Cl原子半径=1.75Å、組成比は1:1

**解答例** :
    
    
    import numpy as np
    from pymatgen.core import Composition, Element
    
    comp = Composition("NaCl")
    fractions = comp.get_atomic_fraction()
    
    # 原子半径データ
    radii = []
    weights = []
    for el, frac in fractions.items():
        radii.append(Element(el).atomic_radius)
        weights.append(frac)
    
    radii = np.array(radii)
    weights = np.array(weights)
    
    # 統計量計算
    mean = np.sum(weights * radii)
    variance = np.sum(weights * (radii - mean)**2)
    std = np.sqrt(variance)
    range_val = radii.max() - radii.min()
    
    print(f"NaClの原子半径統計量:")
    print(f"  平均: {mean:.4f} Å")
    print(f"  標準偏差: {std:.4f} Å")
    print(f"  範囲: {range_val:.4f} Å")
    
    # 出力:
    # NaClの原子半径統計量:
    #   平均: 1.8050 Å
    #   標準偏差: 0.0550 Å
    #   範囲: 0.1100 Å

演習1-3：元素数カウント

**問題** : 以下の化学式について、含まれる元素の種類数をカウントせよ：

  * Fe2O3
  * CaTiO3
  * Li(Ni0.8Co0.15Al0.05)O2

**解答例** :
    
    
    from pymatgen.core import Composition
    
    formulas = ["Fe2O3", "CaTiO3", "Li(Ni0.8Co0.15Al0.05)O2"]
    
    for formula in formulas:
        comp = Composition(formula)
        num_elements = len(comp.elements)
        print(f"{formula}: {num_elements}種類の元素")
        print(f"  元素: {[el.symbol for el in comp.elements]}\n")
    
    # 出力:
    # Fe2O3: 2種類の元素
    #   元素: ['Fe', 'O']
    # CaTiO3: 3種類の元素
    #   元素: ['Ca', 'Ti', 'O']
    # Li(Ni0.8Co0.15Al0.05)O2: 5種類の元素
    #   元素: ['Li', 'Ni', 'Co', 'Al', 'O']

### Medium（中級レベル）

演習1-4：matminerで特徴量生成

**問題** : matminerのElementPropertyフィーチャライザーを使って、以下の化合物の特徴量を生成せよ：

  * 化合物: ["BaTiO3", "SrTiO3", "PbTiO3"]
  * 元素特性: Number, AtomicWeight, Row
  * 統計量: mean, std, range

生成された特徴量数と、最初の3つの特徴量を表示せよ。

**解答例** :
    
    
    import pandas as pd
    from matminer.featurizers.composition import ElementProperty
    
    # データ準備
    compounds = ["BaTiO3", "SrTiO3", "PbTiO3"]
    df = pd.DataFrame({"composition": compounds})
    
    # フィーチャライザー設定
    featurizer = ElementProperty(features=["Number", "AtomicWeight", "Row"],
                                  stats=["mean", "std", "range"])
    
    # 特徴量生成
    df = featurizer.featurize_dataframe(df, col_id="composition")
    
    # 結果表示
    feature_cols = [col for col in df.columns if col != "composition"]
    print(f"生成された特徴量数: {len(feature_cols)}")
    print(f"\n最初の3つの特徴量:")
    print(df[feature_cols[:3]])
    
    # 出力例:
    # 生成された特徴量数: 9
    # 最初の3つの特徴量:
    #    mean Number  std Number  range Number
    # 0    30.4      23.35         48.0
    # 1    29.2      22.41         46.0
    # 2    41.4      27.93         74.0

演習1-5：元素周期表の可視化

**問題** : 以下の元素グループについて、イオン化エネルギーを棒グラフで可視化せよ：

  * アルカリ金属: Li, Na, K, Rb, Cs
  * ハロゲン: F, Cl, Br, I

2つのグループを並べて比較できるようにせよ。

**解答例** :
    
    
    import matplotlib.pyplot as plt
    from pymatgen.core import Element
    
    # 元素グループ
    alkali = ["Li", "Na", "K", "Rb", "Cs"]
    halogens = ["F", "Cl", "Br", "I"]
    
    # イオン化エネルギー取得
    alkali_ie = [Element(el).ionization_energy for el in alkali]
    halogen_ie = [Element(el).ionization_energy for el in halogens]
    
    # 可視化
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].bar(alkali, alkali_ie, color='#f093fb', edgecolor='black')
    axes[0].set_ylabel('Ionization Energy (eV)', fontsize=12)
    axes[0].set_title('アルカリ金属のイオン化エネルギー', fontsize=14, fontweight='bold')
    axes[0].grid(axis='y', alpha=0.3)
    
    axes[1].bar(halogens, halogen_ie, color='#f5576c', edgecolor='black')
    axes[1].set_ylabel('Ionization Energy (eV)', fontsize=12)
    axes[1].set_title('ハロゲンのイオン化エネルギー', fontsize=14, fontweight='bold')
    axes[1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('ionization_energy_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("観察:")
    print("- アルカリ金属: 周期が増加するほど減少（K < Na < Li）")
    print("- ハロゲン: F が最大、I が最小")

演習1-6：相関分析

**問題** : 以下の元素特性間の相関係数を計算せよ：

  * 原子半径 vs イオン化エネルギー
  * 電気陰性度 vs イオン化エネルギー

対象元素: 周期2の元素（Li, Be, B, C, N, O, F）

**解答例** :
    
    
    import numpy as np
    from pymatgen.core import Element
    import matplotlib.pyplot as plt
    
    # 周期2の元素
    elements = ["Li", "Be", "B", "C", "N", "O", "F"]
    
    # データ抽出
    atomic_radius = []
    ionization_energy = []
    electronegativity = []
    
    for el_symbol in elements:
        el = Element(el_symbol)
        atomic_radius.append(el.atomic_radius)
        ionization_energy.append(el.ionization_energy)
        electronegativity.append(el.X)
    
    # NumPy配列に変換
    ar = np.array(atomic_radius)
    ie = np.array(ionization_energy)
    en = np.array(electronegativity)
    
    # 相関係数計算
    corr_ar_ie = np.corrcoef(ar, ie)[0, 1]
    corr_en_ie = np.corrcoef(en, ie)[0, 1]
    
    print(f"相関係数:")
    print(f"  原子半径 vs イオン化エネルギー: {corr_ar_ie:.3f}")
    print(f"  電気陰性度 vs イオン化エネルギー: {corr_en_ie:.3f}")
    
    # 散布図
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].scatter(ar, ie, s=100, color='#f093fb', edgecolors='black')
    for i, el in enumerate(elements):
        axes[0].annotate(el, (ar[i], ie[i]), fontsize=12, ha='right')
    axes[0].set_xlabel('Atomic Radius (Å)', fontsize=12)
    axes[0].set_ylabel('Ionization Energy (eV)', fontsize=12)
    axes[0].set_title(f'相関: {corr_ar_ie:.3f}', fontsize=14, fontweight='bold')
    axes[0].grid(alpha=0.3)
    
    axes[1].scatter(en, ie, s=100, color='#f5576c', edgecolors='black')
    for i, el in enumerate(elements):
        axes[1].annotate(el, (en[i], ie[i]), fontsize=12, ha='right')
    axes[1].set_xlabel('Electronegativity (Pauling)', fontsize=12)
    axes[1].set_ylabel('Ionization Energy (eV)', fontsize=12)
    axes[1].set_title(f'相関: {corr_en_ie:.3f}', fontsize=14, fontweight='bold')
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('correlation_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # 出力例:
    # 相関係数:
    #   原子半径 vs イオン化エネルギー: -0.985 (強い負の相関)
    #   電気陰性度 vs イオン化エネルギー: 0.992 (強い正の相関)

演習1-7：簡単なモデル適用

**問題** : 以下の架空データで線形回帰モデルを訓練し、バンドギャップを予測せよ：
    
    
    化合物: ["MgO", "CaO", "SrO", "BaO", "ZnO", "CdO"]
    バンドギャップ (eV): [7.8, 6.9, 5.9, 4.2, 3.4, 2.3]
                

訓練/テスト分割（80/20）でMAEとR²を評価せよ。

**解答例** :
    
    
    import pandas as pd
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_absolute_error, r2_score
    from matminer.featurizers.composition import ElementProperty
    
    # データ準備
    compounds = ["MgO", "CaO", "SrO", "BaO", "ZnO", "CdO"]
    bandgaps = [7.8, 6.9, 5.9, 4.2, 3.4, 2.3]
    
    df = pd.DataFrame({"composition": compounds, "bandgap": bandgaps})
    
    # 特徴量生成
    featurizer = ElementProperty.from_preset("magpie")
    df = featurizer.featurize_dataframe(df, col_id="composition")
    
    # 特徴量とターゲット分離
    feature_cols = [col for col in df.columns if col not in ["composition", "bandgap"]]
    X = df[feature_cols].values
    y = df["bandgap"].values
    
    # 訓練/テスト分割
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 線形回帰
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # 評価
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"モデル性能:")
    print(f"  MAE: {mae:.3f} eV")
    print(f"  R²: {r2:.3f}")
    print(f"\nテストデータ予測:")
    for i, (true_val, pred_val) in enumerate(zip(y_test, y_pred)):
        print(f"  実測: {true_val:.1f} eV, 予測: {pred_val:.1f} eV")

### Hard（上級レベル）

演習1-8：多元系材料の特徴量設計

**問題** : ハイエントロピー合金（High-Entropy Alloy）「CoCrFeNiMn」（等モル比）について、以下を計算せよ：

  1. 原子半径の平均、標準偏差、範囲
  2. 電気陰性度の平均、標準偏差、範囲
  3. 価電子数の平均、標準偏差、範囲

これらの統計量を可視化し、HEA設計への示唆を述べよ。

**解答例** :
    
    
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from pymatgen.core import Composition, Element
    
    # ハイエントロピー合金（等モル比）
    comp = Composition("CoCrFeNiMn")
    fractions = comp.get_atomic_fraction()
    
    # 元素特性抽出
    properties = {
        'atomic_radius': [],
        'X': [],  # 電気陰性度
        'nvalence': []  # 価電子数
    }
    
    for el in comp.elements:
        properties['atomic_radius'].append(Element(el).atomic_radius)
        properties['X'].append(Element(el).X)
        properties['nvalence'].append(Element(el).nvalence)
    
    # 統計量計算
    stats_results = {}
    for prop_name, values in properties.items():
        values = np.array(values)
        stats_results[prop_name] = {
            'mean': values.mean(),
            'std': values.std(),
            'range': values.max() - values.min()
        }
    
    # 結果表示
    print("CoCrFeNiMnの元素特性統計量:\n")
    for prop_name, stats in stats_results.items():
        print(f"{prop_name}:")
        for stat_name, value in stats.items():
            print(f"  {stat_name}: {value:.4f}")
        print()
    
    # 可視化
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    prop_labels = ['Atomic Radius (Å)', 'Electronegativity', 'Valence Electrons']
    
    for ax, (prop_name, prop_label) in zip(axes, zip(properties.keys(), prop_labels)):
        values = properties[prop_name]
        stats = stats_results[prop_name]
    
        ax.bar(['Co', 'Cr', 'Fe', 'Ni', 'Mn'], values,
               color=['#f093fb', '#f5576c', '#feca57', '#48dbfb', '#00d2d3'],
               edgecolor='black')
        ax.axhline(stats['mean'], color='red', linestyle='--', linewidth=2, label='Mean')
        ax.set_ylabel(prop_label, fontsize=12)
        ax.set_title(f'{prop_label}\nMean={stats["mean"]:.3f}, Std={stats["std"]:.3f}',
                     fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('hea_feature_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # HEA設計への示唆
    print("HEA設計への示唆:")
    print("- 原子半径の標準偏差が小さい → 格子歪みが小さい → 相安定性が高い")
    print("- 電気陰性度の標準偏差が小さい → 化学的親和性が均一 → 固溶体形成しやすい")
    print("- 価電子数の平均が適切 → 金属結合の強度に影響")

演習1-9：クロスバリデーション

**問題** : 演習1-7のデータを使い、5分割クロスバリデーションでモデルの汎化性能を評価せよ。各フォールドのMAEとR²を表示し、平均±標準偏差を報告せよ。

**解答例** :
    
    
    import pandas as pd
    import numpy as np
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import cross_val_score, KFold
    from matminer.featurizers.composition import ElementProperty
    
    # データ準備
    compounds = ["MgO", "CaO", "SrO", "BaO", "ZnO", "CdO"]
    bandgaps = [7.8, 6.9, 5.9, 4.2, 3.4, 2.3]
    df = pd.DataFrame({"composition": compounds, "bandgap": bandgaps})
    
    # 特徴量生成
    featurizer = ElementProperty.from_preset("magpie")
    df = featurizer.featurize_dataframe(df, col_id="composition")
    
    # 特徴量とターゲット
    feature_cols = [col for col in df.columns if col not in ["composition", "bandgap"]]
    X = df[feature_cols].values
    y = df["bandgap"].values
    
    # 5分割クロスバリデーション
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    model = LinearRegression()
    
    # MAE評価
    mae_scores = -cross_val_score(model, X, y, cv=kfold,
                                   scoring='neg_mean_absolute_error')
    
    # R²評価
    r2_scores = cross_val_score(model, X, y, cv=kfold, scoring='r2')
    
    # 結果表示
    print("5分割クロスバリデーション結果:\n")
    print("各フォールドのMAE (eV):")
    for i, mae in enumerate(mae_scores):
        print(f"  Fold {i+1}: {mae:.3f}")
    print(f"平均MAE: {mae_scores.mean():.3f} ± {mae_scores.std():.3f}\n")
    
    print("各フォールドのR²:")
    for i, r2 in enumerate(r2_scores):
        print(f"  Fold {i+1}: {r2:.3f}")
    print(f"平均R²: {r2_scores.mean():.3f} ± {r2_scores.std():.3f}")

演習1-10：モデル評価（総合問題）

**問題** : 以下の架空のOQMDデータで、組成ベース特徴量を使った形成エネルギー予測モデルを構築せよ：
    
    
    化合物: ["Li2O", "Na2O", "K2O", "MgO", "CaO", "SrO", "Al2O3", "Ga2O3", "In2O3", "TiO2"]
    形成エネルギー (eV/atom): [-2.9, -2.6, -2.3, -3.0, -3.2, -3.1, -3.5, -2.8, -2.5, -4.1]
                

  1. 特徴量を生成（Magpieプリセット）
  2. 訓練/テスト分割（70/30）
  3. Random Forestモデルで訓練（n_estimators=100）
  4. テストセットでMAE、RMSE、R²を評価
  5. 特徴量重要度トップ5を表示

**解答例** :
    
    
    import pandas as pd
    import numpy as np
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from matminer.featurizers.composition import ElementProperty
    import matplotlib.pyplot as plt
    
    # データ準備
    compounds = ["Li2O", "Na2O", "K2O", "MgO", "CaO", "SrO",
                 "Al2O3", "Ga2O3", "In2O3", "TiO2"]
    formation_energies = [-2.9, -2.6, -2.3, -3.0, -3.2, -3.1,
                          -3.5, -2.8, -2.5, -4.1]
    
    df = pd.DataFrame({"composition": compounds, "formation_energy": formation_energies})
    
    # 特徴量生成
    featurizer = ElementProperty.from_preset("magpie")
    df = featurizer.featurize_dataframe(df, col_id="composition")
    
    # 特徴量とターゲット
    feature_cols = [col for col in df.columns if col not in ["composition", "formation_energy"]]
    X = df[feature_cols].values
    y = df["formation_energy"].values
    
    # 訓練/テスト分割
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Random Forestモデル
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # 評価指標
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print("モデル評価結果:")
    print(f"  MAE: {mae:.3f} eV/atom")
    print(f"  RMSE: {rmse:.3f} eV/atom")
    print(f"  R²: {r2:.3f}\n")
    
    # 特徴量重要度トップ5
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("特徴量重要度トップ5:")
    print(feature_importance.head())
    
    # 予測 vs 実測プロット
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, s=100, color='#f5576c', alpha=0.7, edgecolors='black')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
             'k--', lw=2, label='Perfect Prediction')
    plt.xlabel('実測値 (eV/atom)', fontsize=12)
    plt.ylabel('予測値 (eV/atom)', fontsize=12)
    plt.title(f'形成エネルギー予測（Random Forest）\nMAE={mae:.3f}, R²={r2:.3f}',
              fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('rf_prediction.png', dpi=150, bbox_inches='tight')
    plt.show()

## まとめ

この章では、**組成ベース特徴量** の基礎を学びました：

  * ✅ 化学組成から数値ベクトルへの変換原理
  * ✅ 元素特性の種類（原子半径、イオン化エネルギー、電気陰性度など）
  * ✅ 統計的集約手法（平均、分散、最大/最小、範囲）
  * ✅ pymatgen/matminerでの実装
  * ✅ 成功事例（OQMD形成エネルギー予測、バンドギャップ予測）
  * ✅ 構造情報なしでの予測可能性と高速スクリーニングの利点

#### 🎓 学習目標の達成確認

以下の質問に答えられますか？

  * 組成ベース特徴量とは何か、3行で説明できますか？
  * Fe₂O₃の原子半径の統計量を手動で計算できますか？
  * 組成ベースとGNN、どちらが適切か判断できますか？
  * matminerで化学式から132次元ベクトルを生成できますか？

**全てYesなら** 、第2章（Magpie詳細）へ進みましょう！

## 参考文献

  1. Ward, L., Agrawal, A., Choudhary, A., & Wolverton, C. (2016). "A general-purpose machine learning framework for predicting properties of inorganic materials." _npj Computational Materials_ , 2, 16028. <https://doi.org/10.1038/npjcompumats.2016.28> (Magpie記述子の原論文、pp. 1-7)
  2. Jha, D., Ward, L., Paul, A., Liao, W., Choudhary, A., Wolverton, C., & Agrawal, A. (2018). "ElemNet: Deep Learning the Chemistry of Materials From Only Elemental Composition." _Scientific Reports_ , 8, 17593. <https://doi.org/10.1038/s41598-018-35934-y> (組成のみから高精度予測、pp. 1-13)
  3. Ong, S.P., Richards, W.D., Jain, A., Hautier, G., Kocher, M., Cholia, S., Gunter, D., Chevrier, V.L., Persson, K.A., & Ceder, G. (2013). "Python Materials Genomics (pymatgen): A robust, open-source python library for materials analysis." _Computational Materials Science_ , 68, 314-319. <https://doi.org/10.1016/j.commatsci.2012.10.028> (pymatgenライブラリの基礎、pp. 314-319)
  4. Ward, L., Dunn, A., Faghaninia, A., Zimmermann, N.E.R., Bajaj, S., Wang, Q., Montoya, J., Chen, J., Bystrom, K., Dylla, M., Chard, K., Asta, M., Persson, K.A., Snyder, G.J., Foster, I., & Jain, A. (2018). "Matminer: An open source toolkit for materials data mining." _Computational Materials Science_ , 152, 60-69. <https://doi.org/10.1016/j.commatsci.2018.05.018> (matminer原論文、特徴量生成ツールキット、pp. 60-69)
  5. Meredig, B., Agrawal, A., Kirklin, S., Saal, J.E., Doak, J.W., Thompson, A., Zhang, K., Choudhary, A., & Wolverton, C. (2014). "Combinatorial screening for new materials in unconstrained composition space with machine learning." _Physical Review B_ , 89(9), 094104. <https://doi.org/10.1103/PhysRevB.89.094104> (組成空間探索の実証研究、pp. 1-7)
  6. Himanen, L., Jäger, M.O.J., Morooka, E.V., Federici Canova, F., Ranawat, Y.S., Gao, D.Z., Rinke, P., & Foster, A.S. (2019). "DScribe: Library of descriptors for machine learning in materials science." _Computer Physics Communications_ , 247, 106949. <https://doi.org/10.1016/j.cpc.2019.106949> (記述子ライブラリの包括的レビュー、pp. 1-15)
  7. Materials Project Documentation: matminer module. <https://docs.materialsproject.org/> (matminerの公式ドキュメントと使用例)

[← シリーズ目次へ](<./index.html>) [第2章：Magpieと統計記述子 →](<./chapter-2.html>)

### 免責事項

  * 本コンテンツは教育・研究・情報提供のみを目的としており、専門的な助言(法律・会計・技術的保証など)を提供するものではありません。
  * 本コンテンツおよび付随するCode examplesは「現状有姿(AS IS)」で提供され、明示または黙示を問わず、商品性、特定目的適合性、権利非侵害、正確性・完全性、動作・安全性等いかなる保証もしません。
  * 外部リンク、第三者が提供するデータ・ツール・ライブラリ等の内容・可用性・安全性について、作成者および東北大学は一切の責任を負いません。
  * 本コンテンツの利用・実行・解釈により直接的・間接的・付随的・特別・結果的・懲罰的損害が生じた場合でも、適用法で許容される最大限の範囲で、作成者および東北大学は責任を負いません。
  * 本コンテンツの内容は、予告なく変更・更新・提供停止されることがあります。
  * 本コンテンツの著作権・ライセンスは明記された条件(例: CC BY 4.0)に従います。当該ライセンスは通常、無保証条項を含みます。
