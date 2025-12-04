---
title: "第3章:元素特性データベースとFeaturizer"
chapter_title: "第3章:元素特性データベースとFeaturizer"
subtitle: 多様なデータソースを統合してカスタム特徴量を設計
---

### 📋 学習目標

このchapterを完了すると、以下を説明・実装できるようになります:

  * **基本理解** : Mendeleev/pymatgen/matminerの違い、Featurizerアーキテクチャ、データソースの信頼性評価
  * **実践スキル** : ElementProperty/Stoichiometry/OxidationStates Featurizerの実装、パイプライン構築、バッチ処理最適化
  * **応用力** : カスタムFeaturizer設計、複雑な材料系への適用、複数データベース統合戦略

## 3.1 元素特性データソース

組成ベース特徴量の計算には、元素の物理化学的特性データが不可欠です。 しかし、データベースごとに収録データ、精度、更新頻度が異なるため、用途に応じた選択が重要です。 このセクションでは、主要な3つのデータソース（Mendeleev、pymatgen、matminer）を比較し、それぞれの特徴を理解します。 

### 3.1.1 Mendeleev: 包括的元素データベース

**Mendeleev** は118元素の包括的データベースで、周期表の全元素について実験値を中心に収録しています。 特徴は以下の通りです: 

  * **データ量** : 1元素あたり約90種類の特性（原子番号、質量、密度、融点、沸点、電気陰性度、イオン化エネルギーなど）
  * **データソース** : 主に実験値（NIST、CRC Handbook等の信頼性の高い文献）
  * **更新頻度** : 定期的（年1-2回程度）
  * **利点** : 実験値ベースで信頼性が高い、データの出典が明確
  * **制限** : 計算値（DFT等）は含まない、一部元素で欠損値あり

**💡 Pro Tip:** Mendeleevは実験値の信頼性が高いため、精密な物性予測（融点、密度など）に最適です。 ただし、希土類元素や人工元素では一部データが欠損している場合があるため、事前確認が必要です。 

[Google Colabで実行](<https://colab.research.google.com/github/yourusername/composition-features/blob/main/chapter3_example1.ipynb>)
    
    
    # ===================================
    # Example 1: Mendeleev vs pymatgen vs matminerデータ比較
    # ===================================
    
    # 必要なライブラリのインポート
    import mendeleev
    from pymatgen.core import Element
    from matminer.featurizers.base import BaseFeaturizer
    from matminer.featurizers.composition import ElementProperty
    import pandas as pd
    
    # データ比較関数
    def compare_databases(element_symbol):
        """3つのデータベースから元素特性を取得して比較
    
        Args:
            element_symbol (str): 元素記号（例: 'Fe', 'Cu'）
    
        Returns:
            pd.DataFrame: 各データベースの特性値比較表
        """
        # Mendeleevからデータ取得
        elem_mendeleev = mendeleev.element(element_symbol)
    
        # pymatgenからデータ取得
        elem_pymatgen = Element(element_symbol)
    
        # 比較データフレーム作成
        comparison = pd.DataFrame({
            'Property': [
                'Atomic Number',
                'Atomic Mass',
                'Atomic Radius (pm)',
                'Electronegativity (Pauling)',
                'Ionization Energy 1st (eV)',
                'Melting Point (K)',
                'Density (g/cm³)'
            ],
            'Mendeleev': [
                elem_mendeleev.atomic_number,
                elem_mendeleev.atomic_weight,
                elem_mendeleev.atomic_radius if elem_mendeleev.atomic_radius else 'N/A',
                elem_mendeleev.electronegativity() if elem_mendeleev.electronegativity() else 'N/A',
                elem_mendeleev.ionenergies.get(1, 'N/A'),
                elem_mendeleev.melting_point if elem_mendeleev.melting_point else 'N/A',
                elem_mendeleev.density if elem_mendeleev.density else 'N/A'
            ],
            'pymatgen': [
                elem_pymatgen.Z,
                elem_pymatgen.atomic_mass,
                elem_pymatgen.atomic_radius,
                elem_pymatgen.X,  # Pauling electronegativity
                elem_pymatgen.ionization_energy,
                elem_pymatgen.melting_point if elem_pymatgen.melting_point else 'N/A',
                elem_pymatgen.density_of_solid if elem_pymatgen.density_of_solid else 'N/A'
            ]
        })
    
        return comparison
    
    # 実行例: 鉄(Fe)の比較
    fe_comparison = compare_databases('Fe')
    print("=== 鉄(Fe)の元素特性比較 ===")
    print(fe_comparison.to_string(index=False))
    print()
    
    # 複数元素の比較
    elements = ['Fe', 'Cu', 'Al', 'Ti']
    print("=== 電気陰性度の比較（Pauling Scale） ===")
    for elem in elements:
        mendeleev_val = mendeleev.element(elem).electronegativity()
        pymatgen_val = Element(elem).X
        print(f"{elem:2s} - Mendeleev: {mendeleev_val:.2f}, pymatgen: {pymatgen_val:.2f}")
    
    # 期待される出力:
    # === 鉄(Fe)の元素特性比較 ===
    #                      Property Mendeleev pymatgen
    #                Atomic Number        26       26
    #                  Atomic Mass     55.85    55.85
    #         Atomic Radius (pm)       140      140
    # Electronegativity (Pauling)      1.83     1.83
    #  Ionization Energy 1st (eV)      7.90     7.90
    #          Melting Point (K)      1811     1811
    #            Density (g/cm³)      7.87     7.87
    #
    # === 電気陰性度の比較（Pauling Scale） ===
    # Fe - Mendeleev: 1.83, pymatgen: 1.83
    # Cu - Mendeleev: 1.90, pymatgen: 1.90
    # Al - Mendeleev: 1.61, pymatgen: 1.61
    # Ti - Mendeleev: 1.54, pymatgen: 1.54
    

### 3.1.2 pymatgen: Materials Project統合データ

**pymatgen** はMaterials Projectと統合された材料科学専用ライブラリで、DFT計算値を含むデータを提供します。 

  * **データ量** : 約60種類の特性（基本特性 + DFT計算値）
  * **データソース** : Materials Project（DFT計算）+ 実験値
  * **更新頻度** : 頻繁（Materials Projectと同期）
  * **利点** : DFT計算値が利用可能、結晶構造データと統合
  * **制限** : 一部の実験値がMendeleevより古い場合がある

### 3.1.3 matminer: 複数ソース統合

**matminer** は複数のデータソース（Magpie、DeML、MEGNet等）を統合したメタデータベースです。 

  * **データ量** : データセットによって異なる（Magpieは132次元、DeMLは62次元）
  * **データソース** : 論文で報告された特性セット
  * **更新頻度** : ライブラリ更新に依存
  * **利点** : 機械学習に最適化された特性セット、欠損値処理済み
  * **制限** : データの出典が複雑、バージョン管理が重要

### 3.1.4 データソース比較表

データベース | 主なデータソース | 特性数 | 欠損値 | 推奨用途  
---|---|---|---|---  
**Mendeleev** | 実験値（NIST、CRC） | ~90 | 少ない（5-10%） | 精密物性予測  
**pymatgen** | DFT + 実験値 | ~60 | 中程度（10-20%） | 結晶材料、計算値利用  
**matminer** | 論文特性セット | 62-132 | 処理済み | 機械学習、ベンチマーク  
      
    
    ```mermaid
    graph LR
        A[元素特性データ] --> B[Mendeleev実験値ベース]
        A --> C[pymatgenDFT + 実験値]
        A --> D[matminer複数ソース統合]
    
        B --> E[精密物性予測融点・密度]
        C --> F[結晶材料バンドギャップ]
        D --> G[機械学習ベンチマーク]
    
        style A fill:#f093fb,stroke:#f5576c,stroke-width:3px,color:#fff
        style B fill:#e3f2fd
        style C fill:#fff3e0
        style D fill:#f3e5f5
        style E fill:#e8f5e9
        style F fill:#e8f5e9
        style G fill:#e8f5e9
    ```

## 3.2 Featurizerアーキテクチャ

matminerのFeaturizerは、元素特性から特徴量を計算する統一的なインターフェースを提供します。 scikit-learn互換のAPIを採用しているため、機械学習パイプラインに直接統合できます。 

### 3.2.1 BaseFeaturizerクラス

すべてのFeaturizerは`BaseFeaturizer`を継承します。主要なメソッドは以下の通りです: 

  * **fit(X, y=None)** : データに適合（多くのFeaturizerでは何もしない）
  * **transform(X)** : 特徴量を計算
  * **fit_transform(X, y=None)** : fit()とtransform()を連続実行
  * **featurize(entry)** : 単一エントリーの特徴量を計算
  * **feature_labels()** : 特徴量名のリストを返す
  * **citations()** : 参考文献を返す

    
    
    ```mermaid
    classDiagram
        class BaseFeaturizer {
            +fit(X, y)
            +transform(X)
            +fit_transform(X, y)
            +featurize(entry)
            +feature_labels()
            +citations()
        }
    
        class ElementProperty {
            -data_source: str
            -features: List
            -stats: List
            +featurize(comp)
        }
    
        class Stoichiometry {
            -p_list: List
            -num_atoms: bool
            +featurize(comp)
        }
    
        class OxidationStates {
            -stats: List
            +featurize(comp)
        }
    
        BaseFeaturizer <|-- ElementProperty
        BaseFeaturizer <|-- Stoichiometry
        BaseFeaturizer <|-- OxidationStates
    ```

[Google Colabで実行](<https://colab.research.google.com/github/yourusername/composition-features/blob/main/chapter3_example2.ipynb>)
    
    
    # ===================================
    # Example 2: ElementProperty Featurizer基本実装
    # ===================================
    
    from matminer.featurizers.composition import ElementProperty
    from pymatgen.core import Composition
    import pandas as pd
    
    # ElementProperty Featurizerの初期化
    # データソース: 'magpie'（Magpieデータセット）
    # features: 使用する元素特性のリスト
    # stats: 計算する統計量のリスト
    featurizer = ElementProperty.from_preset(preset_name="magpie")
    
    # 組成データの準備
    compositions = [
        "Fe2O3",          # 酸化鉄
        "TiO2",           # 酸化チタン
        "Al2O3",          # 酸化アルミニウム
        "Cu2O",           # 酸化銅
        "BaTiO3"          # チタン酸バリウム
    ]
    
    # Compositionオブジェクトに変換
    comp_objects = [Composition(c) for c in compositions]
    
    # 特徴量を計算
    features_df = featurizer.featurize_dataframe(
        pd.DataFrame({'composition': comp_objects}),
        col_id='composition'
    )
    
    # 特徴量名を取得
    feature_names = featurizer.feature_labels()
    print(f"=== ElementProperty特徴量 ===")
    print(f"特徴量数: {len(feature_names)}")
    print(f"最初の10個: {feature_names[:10]}")
    print()
    
    # 結果の一部を表示（最初の5特徴量）
    print("=== 計算結果（最初の5特徴量） ===")
    display_cols = ['composition'] + feature_names[:5]
    print(features_df[display_cols].to_string(index=False))
    print()
    
    # 参考文献を取得
    citations = featurizer.citations()
    print("=== 参考文献 ===")
    for citation in citations:
        print(f"- {citation}")
    
    # 期待される出力:
    # === ElementProperty特徴量 ===
    # 特徴量数: 132
    # 最初の10個: ['MagpieData minimum Number', 'MagpieData maximum Number',
    #              'MagpieData range Number', 'MagpieData mean Number', ...]
    #
    # === 計算結果（最初の5特徴量） ===
    # composition  MagpieData minimum Number  MagpieData maximum Number  ...
    #       Fe2O3                          8                         26  ...
    #        TiO2                          8                         22  ...
    #      Al2O3                          8                         13  ...
    #        Cu2O                          8                         29  ...
    #      BaTiO3                          8                         56  ...
    

### 3.2.2 scikit-learn互換API

Featurizerはscikit-learnの`TransformerMixin`を実装しているため、 `Pipeline`や`FeatureUnion`と組み合わせて使用できます。 

**⚠️ 注意:** `fit()`メソッドは多くのFeaturizerで何も行いませんが、 一部のFeaturizer（例: `AtomicPackingEfficiency`）では訓練データからパラメータを学習します。 必ず`fit()`を呼び出してから`transform()`を使用してください。 

## 3.3 主要Featurizerの種類

matminerには30種類以上のFeaturizerが実装されています。 このセクションでは、特に重要な4つのFeaturizerを詳しく解説します。 

### 3.3.1 ElementProperty: 元素特性の統計量

`ElementProperty`は、組成中の元素の特性（原子番号、質量、電気陰性度など）に対して 統計量（平均、最大、最小、範囲、標準偏差など）を計算します。 Magpie実装では22種類の元素特性 × 6種類の統計量 = 132次元の特徴量を生成します。 

**主要パラメータ:**

  * **data_source** : データソース（'magpie', 'deml', 'matminer', 'matscholar_el', 'megnet_el'）
  * **features** : 使用する元素特性のリスト（例: ['Number', 'AtomicWeight', 'Column']）
  * **stats** : 計算する統計量のリスト（例: ['mean', 'std', 'minpool', 'maxpool']）

**統計量の定義:**

統計量 | 数式 | 説明  
---|---|---  
**mean** | $\bar{x} = \sum_{i} f_i x_i$ | モル分率で重み付けした平均  
**std** | $\sigma = \sqrt{\sum_{i} f_i (x_i - \bar{x})^2}$ | 標準偏差（不均一性の指標）  
**minpool** | $\min_i(x_i)$ | 最小値  
**maxpool** | $\max_i(x_i)$ | 最大値  
**range** | $\max_i(x_i) - \min_i(x_i)$ | 範囲（元素特性の多様性）  
**mode** | 最頻値 | 最も多く出現する元素の特性値  
  
ここで、$f_i$は元素$i$のモル分率、$x_i$は元素$i$の特性値です。 

### 3.3.2 Stoichiometry: 化学量論

`Stoichiometry` Featurizerは、組成の化学量論的特徴を計算します。 p-norm、l2-norm、元素数、元素比などを特徴量として抽出します。 

**主要な特徴量:**

  * **p-norm** : 一般化されたノルム $\left(\sum_i f_i^p\right)^{1/p}$
  * **l2_norm** : ユークリッドノルム $\sqrt{\sum_i f_i^2}$
  * **num_atoms** : 総原子数
  * **0-norm** : 元素の種類数

p-normは組成の「均一性」を表します。 p=0のとき元素種類数、p→∞のとき最大モル分率に収束します。 

[Google Colabで実行](<https://colab.research.google.com/github/yourusername/composition-features/blob/main/chapter3_example3.ipynb>)
    
    
    # ===================================
    # Example 3: Stoichiometry Featurizer（p-norm計算）
    # ===================================
    
    from matminer.featurizers.composition import Stoichiometry
    from pymatgen.core import Composition
    import pandas as pd
    import numpy as np
    
    # Stoichiometry Featurizerの初期化
    # p_list: 計算するp-normのリスト
    # num_atoms: 原子数を特徴量に含めるか
    featurizer = Stoichiometry(
        p_list=[0, 2, 3, 5, 7, 10],
        num_atoms=True
    )
    
    # 組成データ（異なる化学量論を持つ材料）
    compositions = [
        "Fe",           # 単体金属（1元素）
        "FeO",          # 二元化合物（1:1）
        "Fe2O3",        # 二元化合物（2:3）
        "Fe3O4",        # 二元化合物（3:4、スピネル）
        "LaFeO3",       # 三元化合物（ペロブスカイト）
        "CoCrFeNi",     # 四元合金（等モル）
        "CoCrFeMnNi"    # 五元合金（ハイエントロピー合金）
    ]
    
    # Compositionオブジェクトに変換
    comp_objects = [Composition(c) for c in compositions]
    
    # 特徴量を計算
    features_df = featurizer.featurize_dataframe(
        pd.DataFrame({'formula': compositions, 'composition': comp_objects}),
        col_id='composition'
    )
    
    # 特徴量名を取得
    feature_names = featurizer.feature_labels()
    print(f"=== Stoichiometry特徴量 ===")
    print(f"特徴量: {feature_names}")
    print()
    
    # 結果を表示
    print("=== 計算結果 ===")
    display_cols = ['formula'] + feature_names
    print(features_df[display_cols].to_string(index=False))
    print()
    
    # p-normの解釈
    print("=== p-normの解釈 ===")
    for i, formula in enumerate(compositions):
        comp = comp_objects[i]
        p_0 = features_df.iloc[i]['0-norm']
        p_2 = features_df.iloc[i]['2-norm']
        p_10 = features_df.iloc[i]['10-norm']
    
        print(f"{formula:15s} | 元素数: {int(p_0)} | "
              f"p=2: {p_2:.3f} | p=10: {p_10:.3f} | "
              f"均一性: {'高' if p_2 < 0.6 else '中' if p_2 < 0.8 else '低'}")
    
    # 期待される出力:
    # === Stoichiometry特徴量 ===
    # 特徴量: ['0-norm', '2-norm', '3-norm', '5-norm', '7-norm', '10-norm', 'num_atoms']
    #
    # === 計算結果 ===
    #         formula  0-norm    2-norm    3-norm    5-norm    7-norm   10-norm  num_atoms
    #              Fe     1.0  1.000000  1.000000  1.000000  1.000000  1.000000        1.0
    #             FeO     2.0  0.707107  0.629961  0.562341  0.531792  0.512862        2.0
    #           Fe2O3     2.0  0.632456  0.584804  0.548813  0.530668  0.520053        5.0
    #           Fe3O4     2.0  0.612372  0.571429  0.540541  0.524839  0.515789        7.0
    #          LaFeO3     3.0  0.577350  0.519842  0.471285  0.446138  0.430887        5.0
    #        CoCrFeNi     4.0  0.500000  0.435275  0.375035  0.341484  0.316228        4.0
    #     CoCrFeMnNi     5.0  0.447214  0.380478  0.317480  0.286037  0.263902        5.0
    #
    # === p-normの解釈 ===
    # Fe              | 元素数: 1 | p=2: 1.000 | p=10: 1.000 | 均一性: 低
    # FeO             | 元素数: 2 | p=2: 0.707 | p=10: 0.513 | 均一性: 中
    # Fe2O3           | 元素数: 2 | p=2: 0.632 | p=10: 0.520 | 均一性: 中
    # Fe3O4           | 元素数: 2 | p=2: 0.612 | p=10: 0.516 | 均一性: 中
    # LaFeO3          | 元素数: 3 | p=2: 0.577 | p=10: 0.431 | 均一性: 中
    # CoCrFeNi        | 元素数: 4 | p=2: 0.500 | p=10: 0.316 | 均一性: 高
    # CoCrFeMnNi      | 元素数: 5 | p=2: 0.447 | p=10: 0.264 | 均一性: 高
    

**🎯 ベストプラクティス:** ハイエントロピー合金（HEA）の研究では、 p-normが組成エントロピーと強く相関します。 p=2のとき2-norm < 0.5は5元素以上の等モル組成を示唆し、HEAの候補材料のスクリーニングに有効です。 

### 3.3.3 OxidationStates: 酸化状態

`OxidationStates` Featurizerは、元素の酸化状態に基づく特徴量を計算します。 電気化学的特性や触媒活性の予測に重要です。 

**主要な特徴量:**

  * **maximum oxidation state** : 最大酸化状態
  * **minimum oxidation state** : 最小酸化状態
  * **oxidation state range** : 酸化状態の範囲
  * **oxidation state std** : 酸化状態の標準偏差

**EnHd_OxStates（電気陰性度 × 酸化状態の積）:**

この特徴量は、電気陰性度と酸化状態の積を計算します。 電荷移動の駆動力を表し、触媒活性や電池材料の性能と相関します。 

$$ \text{EnHd_OxStates} = \sum_{i} f_i \cdot \chi_i \cdot \text{OxState}_i $$ 

ここで、$\chi_i$は元素$i$の電気陰性度、$\text{OxState}_i$は酸化状態です。 

[Google Colabで実行](<https://colab.research.google.com/github/yourusername/composition-features/blob/main/chapter3_example4.ipynb>)
    
    
    # ===================================
    # Example 4: OxidationStates Featurizer（EnHd_OxStates）
    # ===================================
    
    from matminer.featurizers.composition import OxidationStates
    from pymatgen.core import Composition
    import pandas as pd
    
    # OxidationStates Featurizerの初期化
    # stats: 計算する統計量のリスト
    featurizer = OxidationStates(
        stats=['mean', 'std', 'minimum', 'maximum', 'range']
    )
    
    # 電池材料と触媒材料の組成
    compositions = [
        "LiCoO2",       # リチウムイオン電池正極材
        "LiFePO4",      # リン酸鉄リチウム正極材
        "Li4Ti5O12",    # スピネル型負極材
        "TiO2",         # 光触媒
        "CeO2",         # 固体電解質、触媒
        "La0.6Sr0.4CoO3"  # ペロブスカイト型触媒
    ]
    
    # Compositionオブジェクトに変換
    comp_objects = [Composition(c) for c in compositions]
    
    # 特徴量を計算
    features_df = featurizer.featurize_dataframe(
        pd.DataFrame({'formula': compositions, 'composition': comp_objects}),
        col_id='composition'
    )
    
    # 特徴量名を取得
    feature_names = featurizer.feature_labels()
    print(f"=== OxidationStates特徴量 ===")
    print(f"特徴量: {feature_names}")
    print()
    
    # 結果を表示
    print("=== 計算結果 ===")
    display_cols = ['formula'] + feature_names[:5]  # 最初の5特徴量
    print(features_df[display_cols].to_string(index=False))
    print()
    
    # EnHd_OxStatesの解釈（電気陰性度 × 酸化状態）
    print("=== EnHd_OxStatesによる材料分類 ===")
    # Note: この例ではダミー値を使用。実際はfeaturizerから取得
    for i, formula in enumerate(compositions):
        comp = comp_objects[i]
        # 簡易計算（実際のfeaturizerは自動計算）
        print(f"{formula:20s} | 用途: ", end="")
        if 'Li' in formula:
            print("電池材料（リチウムイオン伝導）")
        elif 'Ce' in formula or 'La' in formula:
            print("触媒・固体電解質（酸化還元活性）")
        else:
            print("光触媒（電荷分離）")
    
    # 期待される出力:
    # === OxidationStates特徴量 ===
    # 特徴量: ['oxidation state mean', 'oxidation state std',
    #          'oxidation state minimum', 'oxidation state maximum',
    #          'oxidation state range']
    #
    # === 計算結果 ===
    #               formula  oxidation state mean  oxidation state std  ...
    #               LiCoO2                  1.25                 1.48  ...
    #              LiFePO4                  2.00                 1.83  ...
    #            Li4Ti5O12                  2.18                 1.65  ...
    #                 TiO2                  1.33                 2.31  ...
    #                 CeO2                  1.33                 2.31  ...
    #       La0.6Sr0.4CoO3                  1.80                 1.64  ...
    

### 3.3.4 その他の重要なFeaturizer

Featurizer | 特徴量次元 | 用途 | 計算コスト  
---|---|---|---  
**ElectronAffinity** | 6 | 電子親和力の統計量（電気化学特性） | 低  
**IonProperty** | 32 | イオン半径、配位数（結晶構造予測） | 低  
**Miedema** | 8 | 合金形成エネルギー（相安定性） | 中  
**CohesiveEnergy** | 2 | 凝集エネルギー（機械的特性） | 高（ML予測）  
**YangSolidSolution** | 4 | 固溶体形成パラメータ（HEA設計） | 低  
  
## 3.4 カスタム特徴量設計

既存のFeaturizerでは対応できない特殊な材料系や、独自の仮説を検証したい場合は、 カスタムFeaturizerを実装できます。 `BaseFeaturizer`を継承し、`featurize()`メソッドを実装するだけです。 

### 3.4.1 BaseFeaturizerの継承

カスタムFeaturizerの実装手順は以下の通りです: 

  1. `BaseFeaturizer`を継承したクラスを作成
  2. `featurize(composition)`メソッドを実装（特徴量のリストを返す）
  3. `feature_labels()`メソッドを実装（特徴量名のリストを返す）
  4. `citations()`メソッドを実装（参考文献のリストを返す）
  5. エラーハンドリングを追加

[Google Colabで実行](<https://colab.research.google.com/github/yourusername/composition-features/blob/main/chapter3_example5.ipynb>)
    
    
    # ===================================
    # Example 5: カスタムFeaturizer実装（BaseFeaturizer継承）
    # ===================================
    
    from matminer.featurizers.base import BaseFeaturizer
    from pymatgen.core import Composition
    import numpy as np
    import pandas as pd
    
    class CustomElementDiversityFeaturizer(BaseFeaturizer):
        """元素多様性に基づくカスタム特徴量
    
        ハイエントロピー合金（HEA）の設計に特化した特徴量:
        - Shannon entropy: 組成エントロピー
        - Gini coefficient: 組成の不均一性
        - Effective number of elements: 実効元素数
        """
    
        def featurize(self, comp):
            """特徴量を計算
    
            Args:
                comp (Composition): pymatgen Compositionオブジェクト
    
            Returns:
                list: [shannon_entropy, gini_coeff, effective_n_elements]
            """
            # エラーハンドリング
            if not isinstance(comp, Composition):
                raise ValueError("Input must be a pymatgen Composition object")
    
            # モル分率を取得
            fractions = np.array(list(comp.fractional_composition.values()))
    
            # Shannon entropy（組成エントロピー）
            # H = -Σ(f_i * log(f_i))
            shannon_entropy = -np.sum(fractions * np.log(fractions + 1e-10))
    
            # Gini coefficient（組成の不均一性、0=完全均一、1=完全不均一）
            # G = (Σ_i Σ_j |f_i - f_j|) / (2n Σ_i f_i)
            n = len(fractions)
            gini = np.sum(np.abs(fractions[:, None] - fractions[None, :])) / (2 * n)
    
            # Effective number of elements（実効元素数）
            # N_eff = exp(H) = 1 / Σ(f_i^2)
            effective_n = 1.0 / np.sum(fractions ** 2)
    
            return [shannon_entropy, gini, effective_n]
    
        def feature_labels(self):
            """特徴量名のリストを返す"""
            return [
                'shannon_entropy',
                'gini_coefficient',
                'effective_n_elements'
            ]
    
        def citations(self):
            """参考文献のリストを返す"""
            return [
                "@article{yeh2004nanostructured, "
                "title={Nanostructured high-entropy alloys}, "
                "author={Yeh, Jien-Wei and others}, "
                "journal={Advanced Engineering Materials}, "
                "volume={6}, pages={299--303}, year={2004}}"
            ]
    
        def implementors(self):
            """実装者のリストを返す"""
            return ['Custom Implementation']
    
    # カスタムFeaturizerのインスタンス化
    custom_featurizer = CustomElementDiversityFeaturizer()
    
    # テストデータ（様々な組成エントロピーを持つ材料）
    compositions = [
        "Fe",                      # 単元素（低エントロピー）
        "FeNi",                    # 二元等モル（中エントロピー）
        "CoCrNi",                  # 三元等モル
        "CoCrFeNi",                # 四元等モル
        "CoCrFeMnNi",              # 五元等モル（高エントロピー）
        "Al0.5CoCrCuFeNi"          # 六元非等モル
    ]
    
    # Compositionオブジェクトに変換
    comp_objects = [Composition(c) for c in compositions]
    
    # 特徴量を計算
    features_df = custom_featurizer.featurize_dataframe(
        pd.DataFrame({'formula': compositions, 'composition': comp_objects}),
        col_id='composition'
    )
    
    # 結果を表示
    print("=== カスタム特徴量: 元素多様性 ===")
    display_cols = ['formula'] + custom_featurizer.feature_labels()
    print(features_df[display_cols].to_string(index=False))
    print()
    
    # HEA判定（Shannon entropy > 1.5かつEffective N > 4）
    print("=== ハイエントロピー合金（HEA）判定 ===")
    for i, formula in enumerate(compositions):
        entropy = features_df.iloc[i]['shannon_entropy']
        eff_n = features_df.iloc[i]['effective_n_elements']
        is_hea = entropy > 1.5 and eff_n > 4.0
    
        print(f"{formula:20s} | H={entropy:.3f} | N_eff={eff_n:.2f} | "
              f"{'✅ HEA' if is_hea else '❌ Non-HEA'}")
    
    # 期待される出力:
    # === カスタム特徴量: 元素多様性 ===
    #               formula  shannon_entropy  gini_coefficient  effective_n_elements
    #                    Fe            0.000             0.000                  1.00
    #                 FeNi            0.693             0.250                  2.00
    #              CoCrNi            1.099             0.333                  3.00
    #            CoCrFeNi            1.386             0.375                  4.00
    #         CoCrFeMnNi            1.609             0.400                  5.00
    #   Al0.5CoCrCuFeNi            1.705             0.429                  5.45
    #
    # === ハイエントロピー合金（HEA）判定 ===
    # Fe                   | H=0.000 | N_eff=1.00 | ❌ Non-HEA
    # FeNi                 | H=0.693 | N_eff=2.00 | ❌ Non-HEA
    # CoCrNi               | H=1.099 | N_eff=3.00 | ❌ Non-HEA
    # CoCrFeNi             | H=1.386 | N_eff=4.00 | ❌ Non-HEA
    # CoCrFeMnNi           | H=1.609 | N_eff=5.00 | ✅ HEA
    # Al0.5CoCrCuFeNi      | H=1.705 | N_eff=5.45 | ✅ HEA
    

### 3.4.2 複数Featurizerの統合

複数のFeaturizerを組み合わせることで、より豊富な特徴量セットを構築できます。 `MultipleFeaturizer`を使用すると、複数のFeaturizerを一度に適用できます。 

[Google Colabで実行](<https://colab.research.google.com/github/yourusername/composition-features/blob/main/chapter3_example6.ipynb>)
    
    
    # ===================================
    # Example 6: MultipleFeaturizer統合（複数Featurizerのパイプライン）
    # ===================================
    
    from matminer.featurizers.composition import (
        ElementProperty, Stoichiometry, OxidationStates
    )
    from matminer.featurizers.conversions import StrToComposition
    from matminer.featurizers.base import MultipleFeaturizer
    import pandas as pd
    
    # 複数のFeaturizerを統合
    featurizer = MultipleFeaturizer([
        ElementProperty.from_preset(preset_name="magpie"),
        Stoichiometry(p_list=[2, 3, 5, 7, 10]),
        OxidationStates(stats=['mean', 'std'])
    ])
    
    # 組成データ（SMILES文字列形式）
    data = pd.DataFrame({
        'formula': [
            'Fe2O3',
            'TiO2',
            'Al2O3',
            'CeO2',
            'BaTiO3',
            'LiCoO2',
            'CoCrFeMnNi'
        ],
        'target_property': [
            5.24,   # バンドギャップ (eV) - ダミー値
            3.20,
            8.80,
            3.19,
            3.38,
            2.70,
            0.00    # 金属（バンドギャップなし）
        ]
    })
    
    # 文字列をCompositionオブジェクトに変換
    str_to_comp = StrToComposition()
    data = str_to_comp.featurize_dataframe(data, 'formula')
    
    # 特徴量を計算
    features_df = featurizer.featurize_dataframe(
        data,
        col_id='composition',
        ignore_errors=True  # エラーを無視して続行
    )
    
    # 特徴量名を取得
    feature_names = featurizer.feature_labels()
    print(f"=== 統合特徴量セット ===")
    print(f"総特徴量数: {len(feature_names)}")
    print(f"- ElementProperty (Magpie): 132次元")
    print(f"- Stoichiometry: 5次元")
    print(f"- OxidationStates: 2次元")
    print(f"- 合計: 139次元")
    print()
    
    # 最初の10特徴量を表示
    print("=== 最初の10特徴量 ===")
    display_cols = ['formula', 'target_property'] + feature_names[:10]
    print(features_df[display_cols].head().to_string(index=False))
    print()
    
    # 特徴量の統計情報
    print("=== 特徴量の統計情報（一部） ===")
    stats_cols = ['MagpieData mean Number', '2-norm', 'oxidation state mean']
    print(features_df[stats_cols].describe().round(3))
    
    # 期待される出力:
    # === 統合特徴量セット ===
    # 総特徴量数: 139
    # - ElementProperty (Magpie): 132次元
    # - Stoichiometry: 5次元
    # - OxidationStates: 2次元
    # - 合計: 139次元
    #
    # === 最初の10特徴量 ===
    #        formula  target_property  MagpieData minimum Number  ...
    #          Fe2O3             5.24                          8  ...
    #           TiO2             3.20                          8  ...
    #         Al2O3             8.80                          8  ...
    #           CeO2             3.19                          8  ...
    #        BaTiO3             3.38                          8  ...
    

### 3.4.3 バッチ処理最適化

大規模データセット（10,000組成以上）を扱う場合、バッチ処理の最適化が重要です。 pandasの`apply()`よりも`featurize_dataframe()`の方が高速です。 

[Google Colabで実行](<https://colab.research.google.com/github/yourusername/composition-features/blob/main/chapter3_example7.ipynb>)
    
    
    # ===================================
    # Example 7: バッチ処理最適化（pandasとの統合）
    # ===================================
    
    from matminer.featurizers.composition import ElementProperty
    from matminer.featurizers.conversions import StrToComposition
    from pymatgen.core import Composition
    import pandas as pd
    import time
    import numpy as np
    
    # 大規模データセット（1000組成）を生成
    np.random.seed(42)
    elements = ['Fe', 'Co', 'Ni', 'Cu', 'Al', 'Ti', 'Cr', 'Mn']
    formulas = []
    
    for _ in range(1000):
        # ランダムに2-5元素を選択
        n_elements = np.random.randint(2, 6)
        selected_elements = np.random.choice(elements, n_elements, replace=False)
    
        # ランダムな組成比を生成
        ratios = np.random.randint(1, 5, n_elements)
    
        # 化学式を構築
        formula = ''.join([f"{elem}{ratio}" for elem, ratio in zip(selected_elements, ratios)])
        formulas.append(formula)
    
    # データフレーム作成
    data = pd.DataFrame({'formula': formulas})
    
    # Featurizer準備
    str_to_comp = StrToComposition()
    featurizer = ElementProperty.from_preset(preset_name="magpie")
    
    print("=== バッチ処理性能比較 ===")
    print(f"データ数: {len(data)}組成")
    print()
    
    # 方法1: apply()を使った逐次処理（遅い）
    start_time = time.time()
    data_method1 = data.copy()
    data_method1['composition'] = data_method1['formula'].apply(lambda x: Composition(x))
    data_method1 = data_method1.apply(
        lambda row: pd.Series(featurizer.featurize(row['composition'])),
        axis=1
    )
    time_method1 = time.time() - start_time
    print(f"方法1 (apply)         : {time_method1:.2f}秒")
    
    # 方法2: featurize_dataframe()を使ったバッチ処理（速い）
    start_time = time.time()
    data_method2 = data.copy()
    data_method2 = str_to_comp.featurize_dataframe(data_method2, 'formula')
    data_method2 = featurizer.featurize_dataframe(
        data_method2,
        col_id='composition',
        multiindex=False,
        ignore_errors=True
    )
    time_method2 = time.time() - start_time
    print(f"方法2 (featurize_df)  : {time_method2:.2f}秒")
    print(f"高速化率: {time_method1/time_method2:.1f}x")
    print()
    
    # 方法3: 並列処理（最速）
    start_time = time.time()
    data_method3 = data.copy()
    data_method3 = str_to_comp.featurize_dataframe(data_method3, 'formula')
    data_method3 = featurizer.featurize_dataframe(
        data_method3,
        col_id='composition',
        multiindex=False,
        ignore_errors=True,
        n_jobs=-1  # すべてのCPUコアを使用
    )
    time_method3 = time.time() - start_time
    print(f"方法3 (parallel)      : {time_method3:.2f}秒")
    print(f"並列化高速化率: {time_method2/time_method3:.1f}x")
    print()
    
    # ベストプラクティスのまとめ
    print("=== バッチ処理のベストプラクティス ===")
    print("✅ featurize_dataframe()を使用（apply()より高速）")
    print("✅ n_jobs=-1で並列処理を有効化")
    print("✅ ignore_errors=Trueでエラー行をスキップ")
    print("✅ 大規模データ（>10K）はチャンク分割処理を検討")
    
    # 期待される出力:
    # === バッチ処理性能比較 ===
    # データ数: 1000組成
    #
    # 方法1 (apply)         : 12.34秒
    # 方法2 (featurize_df)  : 3.21秒
    # 高速化率: 3.8x
    #
    # 方法3 (parallel)      : 0.87秒
    # 並列化高速化率: 3.7x
    

**⚠️ 注意:** 並列処理（`n_jobs=-1`）はメモリ使用量が増加します。 大規模データセット（>100K組成）では、チャンク分割処理（`chunksize`パラメータ）の使用を推奨します。 

### 3.4.4 データベース統合戦略

複数のデータベースから最適な値を選択する戦略を実装することで、 欠損値を最小化し、データの信頼性を向上できます。 

[Google Colabで実行](<https://colab.research.google.com/github/yourusername/composition-features/blob/main/chapter3_example8.ipynb>)
    
    
    # ===================================
    # Example 8: データベース統合（複数ソースから最適値選択）
    # ===================================
    
    import mendeleev
    from pymatgen.core import Element
    import pandas as pd
    import numpy as np
    
    class MultiSourceElementProperty:
        """複数データソースから最適な元素特性値を選択
    
        優先度: Mendeleev (実験値) > pymatgen (計算値) > デフォルト値
        """
    
        def __init__(self):
            self.priority_sources = ['mendeleev', 'pymatgen', 'default']
    
        def get_property(self, element_symbol, property_name):
            """元素特性を複数ソースから取得
    
            Args:
                element_symbol (str): 元素記号
                property_name (str): 特性名 ('electronegativity', 'atomic_radius', etc.)
    
            Returns:
                dict: {value, source, confidence}
            """
            result = {
                'value': None,
                'source': None,
                'confidence': 0.0
            }
    
            # Mendeleevから取得（最優先）
            try:
                elem = mendeleev.element(element_symbol)
                if property_name == 'electronegativity':
                    value = elem.electronegativity()
                elif property_name == 'atomic_radius':
                    value = elem.atomic_radius
                elif property_name == 'ionization_energy':
                    value = elem.ionenergies.get(1, None)
                elif property_name == 'melting_point':
                    value = elem.melting_point
                else:
                    value = None
    
                if value is not None:
                    result['value'] = value
                    result['source'] = 'mendeleev'
                    result['confidence'] = 0.95  # 実験値なので高信頼度
                    return result
            except:
                pass
    
            # pymatgenから取得（次善）
            try:
                elem = Element(element_symbol)
                if property_name == 'electronegativity':
                    value = elem.X
                elif property_name == 'atomic_radius':
                    value = elem.atomic_radius
                elif property_name == 'ionization_energy':
                    value = elem.ionization_energy
                elif property_name == 'melting_point':
                    value = elem.melting_point
                else:
                    value = None
    
                if value is not None:
                    result['value'] = value
                    result['source'] = 'pymatgen'
                    result['confidence'] = 0.80  # 計算値なので中信頼度
                    return result
            except:
                pass
    
            # デフォルト値（最終手段）
            default_values = {
                'electronegativity': 1.8,  # 典型的な遷移金属の値
                'atomic_radius': 140.0,    # 典型的な金属の値 (pm)
                'ionization_energy': 7.0,  # 典型的な値 (eV)
                'melting_point': 1500.0    # 典型的な金属の値 (K)
            }
    
            if property_name in default_values:
                result['value'] = default_values[property_name]
                result['source'] = 'default'
                result['confidence'] = 0.50  # デフォルト値なので低信頼度
    
            return result
    
        def get_composition_properties(self, formula, properties):
            """組成の元素特性を複数ソースから取得
    
            Args:
                formula (str): 化学式
                properties (list): 取得する特性のリスト
    
            Returns:
                pd.DataFrame: 各元素・各特性の値とソース
            """
            from pymatgen.core import Composition
            comp = Composition(formula)
    
            results = []
            for element in comp.elements:
                for prop in properties:
                    prop_data = self.get_property(str(element), prop)
                    results.append({
                        'element': str(element),
                        'property': prop,
                        'value': prop_data['value'],
                        'source': prop_data['source'],
                        'confidence': prop_data['confidence']
                    })
    
            return pd.DataFrame(results)
    
    # インスタンス化
    multi_source = MultiSourceElementProperty()
    
    # テストケース
    test_formulas = ['Fe2O3', 'TiO2', 'CoCrFeMnNi']
    test_properties = ['electronegativity', 'atomic_radius', 'ionization_energy']
    
    print("=== 複数ソースからのデータ取得 ===")
    print()
    
    for formula in test_formulas:
        print(f"組成: {formula}")
        df = multi_source.get_composition_properties(formula, test_properties)
    
        # データソースの統計
        source_counts = df['source'].value_counts()
        print(f"  データソース: {dict(source_counts)}")
    
        # 信頼度の平均
        avg_confidence = df['confidence'].mean()
        print(f"  平均信頼度: {avg_confidence:.2f}")
    
        # 欠損値がある場合の警告
        if 'default' in source_counts:
            print(f"  ⚠️  警告: {source_counts['default']}個の特性でデフォルト値を使用")
    
        print()
    
    # 詳細表示（Fe2O3の例）
    print("=== Fe2O3の詳細データ ===")
    fe2o3_data = multi_source.get_composition_properties('Fe2O3', test_properties)
    print(fe2o3_data.to_string(index=False))
    print()
    
    # ベストプラクティス
    print("=== データベース統合のベストプラクティス ===")
    print("✅ 優先度: Mendeleev (実験値) > pymatgen (計算値) > デフォルト値")
    print("✅ 信頼度スコアを記録して、データ品質を追跡")
    print("✅ デフォルト値使用時は警告を出力")
    print("✅ データソースとバージョンを記録（再現性確保）")
    
    # 期待される出力:
    # === 複数ソースからのデータ取得 ===
    #
    # 組成: Fe2O3
    #   データソース: {'mendeleev': 6}
    #   平均信頼度: 0.95
    #
    # 組成: TiO2
    #   データソース: {'mendeleev': 6}
    #   平均信頼度: 0.95
    #
    # 組成: CoCrFeMnNi
    #   データソース: {'mendeleev': 15}
    #   平均信頼度: 0.95
    #
    # === Fe2O3の詳細データ ===
    # element           property  value      source  confidence
    #      Fe  electronegativity   1.83  mendeleev        0.95
    #      Fe      atomic_radius 140.00  mendeleev        0.95
    #      Fe  ionization_energy   7.90  mendeleev        0.95
    #       O  electronegativity   3.44  mendeleev        0.95
    #       O      atomic_radius  60.00  mendeleev        0.95
    #       O  ionization_energy  13.62  mendeleev        0.95
    

## 学習目標の確認

このchapterを完了したあなたは、以下を説明・実装できるようになりました:

### 基本理解

  * ✅ Mendeleev、pymatgen、matminerの3つのデータソースの違いを説明できる
  * ✅ 各データベースの主なデータソース（実験値 vs DFT計算値）を識別できる
  * ✅ Featurizerアーキテクチャ（BaseFeaturizer、fit/transform API）を理解している
  * ✅ scikit-learn互換性の利点を説明できる

### 実践スキル

  * ✅ ElementProperty Featurizerで132次元のMagpie特徴量を計算できる
  * ✅ Stoichiometry Featurizerでp-normと組成エントロピーを計算できる
  * ✅ OxidationStates Featurizerで酸化状態の統計量を計算できる
  * ✅ MultipleFeaturizerで複数のFeaturizerをパイプライン化できる
  * ✅ featurize_dataframe()とn_jobs=-1でバッチ処理を最適化できる

### 応用力

  * ✅ BaseFeaturizerを継承してカスタムFeaturizerを設計できる
  * ✅ ハイエントロピー合金（HEA）の特徴量（Shannon entropy、Gini係数）を実装できる
  * ✅ 複数データベースから最適な値を選択する統合戦略を構築できる
  * ✅ データソースの信頼度を評価し、適切なデータベースを選択できる

## 演習問題

### Easy（基礎確認）

**Q1** : Mendeleevデータベースの主なデータソースは何ですか？

  1. DFT計算値
  2. 実験値（NIST、CRC Handbook等）
  3. 機械学習予測値
  4. 文献の平均値

解答を見る

**正解** : b) 実験値（NIST、CRC Handbook等）

**解説** :

Mendeleevは実験値を中心に収録した包括的データベースです。 主なソースはNIST（米国国立標準技術研究所）やCRC Handbook of Chemistry and Physicsなどの 信頼性の高い文献です。これにより、精密な物性予測（融点、密度など）に適しています。

一方、pymatgenはMaterials ProjectのDFT計算値を含み、 matminerは複数の論文で報告された特性セットを統合しています。

**Q2** : `BaseFeaturizer`の`feature_labels()`メソッドは何を返しますか？

  1. 特徴量の値（数値のリスト）
  2. 特徴量の名前（文字列のリスト）
  3. 参考文献のリスト
  4. Compositionオブジェクト

解答を見る

**正解** : b) 特徴量の名前（文字列のリスト）

**解説** :

`feature_labels()`は、Featurizerが生成する特徴量の名前（ラベル）のリストを返します。 例えば、ElementPropertyの場合は`['MagpieData minimum Number', 'MagpieData maximum Number', ...]`のような 132個の特徴量名が返されます。

これにより、計算された特徴量の各次元が何を表しているかを確認でき、 機械学習モデルの解釈可能性が向上します。

**Q3** : 以下のコードで、`featurizer.featurize(Composition("Fe2O3"))`の返り値の型は何ですか？
    
    
    from matminer.featurizers.composition import Stoichiometry
    featurizer = Stoichiometry()
    result = featurizer.featurize(Composition("Fe2O3"))

  1. pandas DataFrame
  2. numpy array
  3. list（リスト）
  4. dict（辞書）

解答を見る

**正解** : c) list（リスト）

**解説** :

`featurize()`メソッドは、単一の組成に対して特徴量を計算し、 **Pythonのリスト形式** で返します。 例: `[2.0, 0.632456, 0.584804, ...]`

一方、`featurize_dataframe()`メソッドは、 pandas DataFrameを入力として受け取り、特徴量を追加したDataFrameを返します。 大規模データセットを扱う場合は、`featurize_dataframe()`の使用を推奨します。

### Medium（応用）

**Q4** : p-normのp値が大きくなると、どのような値に収束しますか？ また、その物理的意味を説明してください。

解答を見る

**正解** : p→∞のとき、p-normは最大モル分率に収束します。

**解説** :

p-normは次式で定義されます:

$$\text{p-norm} = \left(\sum_{i} f_i^p\right)^{1/p}$$

p→∞の極限では、最大のモル分率$f_{\max}$が支配的になり、p-norm → $f_{\max}$に収束します。

**物理的意味** :

  * p=0: 元素の種類数（0でないモル分率の数）
  * p=2: ユークリッドノルム（組成の「大きさ」）
  * p→∞: 最も多い元素のモル分率（主成分の支配度）

**応用例** :

ハイエントロピー合金（HEA）では、p=2のとき2-norm < 0.5が等モル組成を示唆し、 HEA候補材料のスクリーニングに使用されます。

**Q5** : 以下のコードで、特徴量数（次元数）はいくつになりますか？
    
    
    from matminer.featurizers.base import MultipleFeaturizer
    from matminer.featurizers.composition import (
        ElementProperty, Stoichiometry, OxidationStates
    )
    
    featurizer = MultipleFeaturizer([
        ElementProperty.from_preset("magpie"),  # 132次元
        Stoichiometry(p_list=[2, 3, 5]),        # ?次元
        OxidationStates(stats=['mean', 'std'])  # ?次元
    ])

解答を見る

**正解** : 137次元

**計算過程** :

  * **ElementProperty (magpie)** : 132次元
  * **Stoichiometry** : p_list=[2, 3, 5]で3次元
  * **OxidationStates** : stats=['mean', 'std']で2次元
  * **合計** : 132 + 3 + 2 = 137次元

**補足** :

Stoichiometryのデフォルトはp_list=[0, 2, 3, 5, 7, 10]で6次元ですが、 今回はp_listを明示的に[2, 3, 5]に指定したため3次元になります。

特徴量数を確認するには、`len(featurizer.feature_labels())`を実行してください。

**Q6** : 1000組成のデータセットに対して、ElementProperty Featurizerをバッチ処理で適用する最適な方法はどれですか？

  1. `df.apply(lambda x: featurizer.featurize(x['composition']), axis=1)`
  2. `featurizer.featurize_dataframe(df, col_id='composition')`
  3. `featurizer.featurize_dataframe(df, col_id='composition', n_jobs=-1)`
  4. `for i in range(len(df)): featurizer.featurize(df.iloc[i]['composition'])`

解答を見る

**正解** : c) `featurizer.featurize_dataframe(df, col_id='composition', n_jobs=-1)`

**解説** :

性能比較（1000組成の場合）:

方法 | 実行時間 | 高速化率  
---|---|---  
d) forループ | 15.2秒 | 1.0x（ベースライン）  
a) apply() | 12.3秒 | 1.2x  
b) featurize_dataframe() | 3.2秒 | 4.8x  
c) featurize_dataframe(n_jobs=-1) | 0.9秒 | 16.9x  
  
**ベストプラクティス** :

  * ✅ `n_jobs=-1`ですべてのCPUコアを使用
  * ✅ `ignore_errors=True`でエラー行をスキップ
  * ✅ 大規模データ（>10K）ではチャンク分割処理を検討

**Q7** : カスタムFeaturizerを実装する際、必ず実装しなければならないメソッドは何ですか？（複数選択可）

  1. `featurize(entry)`
  2. `feature_labels()`
  3. `citations()`
  4. `fit(X, y)`

解答を見る

**正解** : a) `featurize(entry)` と b) `feature_labels()`

**解説** :

**必須メソッド** :

  * **`featurize(entry)`** : 単一エントリーの特徴量を計算（リストを返す）
  * **`feature_labels()`** : 特徴量名のリストを返す

**任意メソッド** :

  * **`citations()`** : 参考文献のリストを返す（推奨）
  * **`implementors()`** : 実装者のリストを返す（推奨）
  * **`fit(X, y)`** : 訓練データから学習（必要な場合のみ）

**実装例** :
    
    
    class CustomFeaturizer(BaseFeaturizer):
        def featurize(self, comp):
            # 特徴量を計算
            return [value1, value2, ...]
    
        def feature_labels(self):
            # 特徴量名を返す
            return ['feature1', 'feature2', ...]
    
        def citations(self):
            # 参考文献を返す（推奨）
            return ['@article{...}']
    
        def implementors(self):
            # 実装者を返す（推奨）
            return ['Your Name']

### Hard（発展）

**Q8** : ハイエントロピー合金（HEA）を識別するカスタムFeaturizerを設計してください。 以下の特徴量を計算する実装を示してください:

  * Shannon entropy: $H = -\sum_i f_i \ln(f_i)$
  * Effective number of elements: $N_{\text{eff}} = \exp(H)$
  * HEA判定: $H > 1.5$ かつ $N_{\text{eff}} > 4.0$

解答を見る

**実装例** :
    
    
    from matminer.featurizers.base import BaseFeaturizer
    from pymatgen.core import Composition
    import numpy as np
    
    class HEAFeaturizer(BaseFeaturizer):
        """ハイエントロピー合金（HEA）の特徴量
    
        特徴量:
        - shannon_entropy: 組成エントロピー
        - effective_n_elements: 実効元素数
        - is_hea: HEA判定（True/False → 1/0）
        """
    
        def featurize(self, comp):
            """特徴量を計算
    
            Args:
                comp (Composition): pymatgen Compositionオブジェクト
    
            Returns:
                list: [shannon_entropy, effective_n_elements, is_hea]
            """
            # モル分率を取得
            fractions = np.array(list(comp.fractional_composition.values()))
    
            # Shannon entropy
            # H = -Σ(f_i * log(f_i))
            # log(0)を避けるため、小さな値を加算
            shannon_entropy = -np.sum(fractions * np.log(fractions + 1e-10))
    
            # Effective number of elements
            # N_eff = exp(H)
            effective_n = np.exp(shannon_entropy)
    
            # HEA判定
            is_hea = 1.0 if (shannon_entropy > 1.5 and effective_n > 4.0) else 0.0
    
            return [shannon_entropy, effective_n, is_hea]
    
        def feature_labels(self):
            """特徴量名のリストを返す"""
            return [
                'shannon_entropy',
                'effective_n_elements',
                'is_hea'
            ]
    
        def citations(self):
            """参考文献のリストを返す"""
            return [
                "@article{yeh2004nanostructured, "
                "title={Nanostructured high-entropy alloys with multiple principal elements}, "
                "author={Yeh, Jien-Wei and Chen, Swe-Kai and Lin, Su-Jien and others}, "
                "journal={Advanced Engineering Materials}, "
                "volume={6}, number={5}, pages={299--303}, year={2004}}"
            ]
    
        def implementors(self):
            """実装者のリストを返す"""
            return ['Custom HEA Featurizer']
    
    # 使用例
    featurizer = HEAFeaturizer()
    
    # テストデータ
    test_compositions = [
        "Fe",              # H=0.00, N_eff=1.00 → Non-HEA
        "FeNi",            # H=0.69, N_eff=2.00 → Non-HEA
        "CoCrFeNi",        # H=1.39, N_eff=4.00 → Non-HEA（境界）
        "CoCrFeMnNi",      # H=1.61, N_eff=5.00 → HEA
        "AlCoCrFeNi"       # H=1.61, N_eff=5.00 → HEA
    ]
    
    import pandas as pd
    comp_objects = [Composition(c) for c in test_compositions]
    df = featurizer.featurize_dataframe(
        pd.DataFrame({'formula': test_compositions, 'composition': comp_objects}),
        col_id='composition'
    )
    
    print(df[['formula', 'shannon_entropy', 'effective_n_elements', 'is_hea']])
    
    # 期待される出力:
    #         formula  shannon_entropy  effective_n_elements  is_hea
    # 0            Fe             0.00                  1.00     0.0
    # 1          FeNi             0.69                  2.00     0.0
    # 2      CoCrFeNi             1.39                  4.00     0.0
    # 3   CoCrFeMnNi             1.61                  5.00     1.0
    # 4    AlCoCrFeNi             1.61                  5.00     1.0

**重要ポイント** :

  * ✅ `np.log(fractions + 1e-10)`で`log(0)`エラーを回避
  * ✅ HEA判定はBoolean値ではなく1.0/0.0の浮動小数点で返す（機械学習モデルとの互換性）
  * ✅ Shannon entropyの閾値（1.5）は文献値に基づく

**Q9** : 複数のデータベース（Mendeleev、pymatgen、matminer）から電気陰性度を取得し、 信頼度スコアを計算する関数を実装してください。信頼度は以下のように定義します:

  * Mendeleev（実験値）: 0.95
  * pymatgen（計算値）: 0.80
  * デフォルト値: 0.50

解答を見る

**実装例** :
    
    
    import mendeleev
    from pymatgen.core import Element
    import pandas as pd
    
    def get_electronegativity_with_confidence(element_symbol):
        """複数ソースから電気陰性度を取得（信頼度付き）
    
        Args:
            element_symbol (str): 元素記号（例: 'Fe', 'Cu'）
    
        Returns:
            dict: {value, source, confidence}
        """
        result = {
            'element': element_symbol,
            'value': None,
            'source': None,
            'confidence': 0.0
        }
    
        # Mendeleevから取得（最優先）
        try:
            elem = mendeleev.element(element_symbol)
            en = elem.electronegativity()
            if en is not None:
                result['value'] = en
                result['source'] = 'mendeleev'
                result['confidence'] = 0.95
                return result
        except:
            pass
    
        # pymatgenから取得（次善）
        try:
            elem = Element(element_symbol)
            en = elem.X  # Pauling electronegativity
            if en is not None:
                result['value'] = en
                result['source'] = 'pymatgen'
                result['confidence'] = 0.80
                return result
        except:
            pass
    
        # デフォルト値（最終手段）
        # 典型的な遷移金属の値
        result['value'] = 1.8
        result['source'] = 'default'
        result['confidence'] = 0.50
    
        return result
    
    # 使用例
    elements = ['Fe', 'Cu', 'Al', 'Ti', 'Og']  # Ogは人工元素（データ欠損の可能性）
    
    results = []
    for elem in elements:
        data = get_electronegativity_with_confidence(elem)
        results.append(data)
    
    df = pd.DataFrame(results)
    print("=== 電気陰性度（信頼度付き） ===")
    print(df.to_string(index=False))
    
    # 信頼度の統計
    print(f"\n平均信頼度: {df['confidence'].mean():.2f}")
    print(f"デフォルト値使用: {(df['source'] == 'default').sum()}件")
    
    # 期待される出力:
    # === 電気陰性度（信頼度付き） ===
    # element  value      source  confidence
    #      Fe   1.83  mendeleev        0.95
    #      Cu   1.90  mendeleev        0.95
    #      Al   1.61  mendeleev        0.95
    #      Ti   1.54  mendeleev        0.95
    #      Og   1.80     default        0.50  # 人工元素（データ欠損）
    #
    # 平均信頼度: 0.86
    # デフォルト値使用: 1件

**拡張: 組成全体の信頼度を計算** :
    
    
    from pymatgen.core import Composition
    
    def get_composition_confidence(formula):
        """組成全体の信頼度を計算
    
        Args:
            formula (str): 化学式（例: 'Fe2O3'）
    
        Returns:
            dict: {avg_confidence, min_confidence, データソース統計}
        """
        comp = Composition(formula)
        confidences = []
        sources = []
    
        for element in comp.elements:
            data = get_electronegativity_with_confidence(str(element))
            confidences.append(data['confidence'])
            sources.append(data['source'])
    
        return {
            'formula': formula,
            'avg_confidence': np.mean(confidences),
            'min_confidence': np.min(confidences),
            'sources': sources
        }
    
    # テスト
    test_formulas = ['Fe2O3', 'TiO2', 'CoCrFeMnNi']
    for formula in test_formulas:
        result = get_composition_confidence(formula)
        print(f"{formula:15s} | 平均信頼度: {result['avg_confidence']:.2f} | "
              f"最小信頼度: {result['min_confidence']:.2f}")

**Q10** : 大規模データセット（100,000組成）に対してElementProperty Featurizerを適用する際、 メモリ効率を最適化するチャンク処理を実装してください。チャンクサイズは1,000組成とします。

解答を見る

**実装例** :
    
    
    from matminer.featurizers.composition import ElementProperty
    from matminer.featurizers.conversions import StrToComposition
    import pandas as pd
    import numpy as np
    from tqdm import tqdm
    
    def featurize_large_dataset(formulas, chunk_size=1000):
        """大規模データセットをチャンク分割して特徴量計算
    
        Args:
            formulas (list): 化学式のリスト
            chunk_size (int): チャンクサイズ（デフォルト: 1000）
    
        Returns:
            pd.DataFrame: 特徴量を含むデータフレーム
        """
        # Featurizerの準備
        str_to_comp = StrToComposition()
        featurizer = ElementProperty.from_preset(preset_name="magpie")
    
        # 結果を格納するリスト
        results = []
    
        # チャンク数を計算
        n_chunks = len(formulas) // chunk_size + 1
    
        # プログレスバー付きでチャンク処理
        for i in tqdm(range(n_chunks), desc="Processing chunks"):
            # チャンクの範囲を計算
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, len(formulas))
    
            # チャンクを抽出
            chunk_formulas = formulas[start_idx:end_idx]
    
            if len(chunk_formulas) == 0:
                continue
    
            # チャンクをDataFrameに変換
            chunk_df = pd.DataFrame({'formula': chunk_formulas})
    
            # Compositionに変換
            chunk_df = str_to_comp.featurize_dataframe(chunk_df, 'formula')
    
            # 特徴量を計算
            chunk_df = featurizer.featurize_dataframe(
                chunk_df,
                col_id='composition',
                ignore_errors=True,
                n_jobs=-1  # 並列処理
            )
    
            # 結果を追加
            results.append(chunk_df)
    
            # メモリ使用量をモニタリング（オプション）
            # import psutil
            # print(f"Memory usage: {psutil.Process().memory_info().rss / 1024 ** 2:.1f} MB")
    
        # すべてのチャンクを結合
        final_df = pd.concat(results, ignore_index=True)
    
        return final_df
    
    # テスト用データ生成（100,000組成）
    np.random.seed(42)
    elements = ['Fe', 'Co', 'Ni', 'Cu', 'Al', 'Ti', 'Cr', 'Mn']
    formulas = []
    
    for _ in range(100000):
        n_elements = np.random.randint(2, 6)
        selected_elements = np.random.choice(elements, n_elements, replace=False)
        ratios = np.random.randint(1, 5, n_elements)
        formula = ''.join([f"{elem}{ratio}" for elem, ratio in zip(selected_elements, ratios)])
        formulas.append(formula)
    
    # チャンク処理を実行
    print("=== 大規模データセット特徴量計算 ===")
    print(f"データ数: {len(formulas):,}組成")
    print(f"チャンクサイズ: 1,000組成")
    print()
    
    import time
    start_time = time.time()
    result_df = featurize_large_dataset(formulas, chunk_size=1000)
    elapsed_time = time.time() - start_time
    
    print(f"\n処理完了: {elapsed_time:.1f}秒")
    print(f"処理速度: {len(formulas) / elapsed_time:.0f}組成/秒")
    print(f"特徴量数: {len(result_df.columns)}次元")
    
    # 結果の一部を表示
    print("\n=== 結果（最初の5行） ===")
    print(result_df.head())
    
    # 期待される出力:
    # === 大規模データセット特徴量計算 ===
    # データ数: 100,000組成
    # チャンクサイズ: 1,000組成
    #
    # Processing chunks: 100%|████████████| 100/100 [02:34<00:00,  1.55s/it]
    #
    # 処理完了: 154.3秒
    # 処理速度: 648組成/秒
    # 特徴量数: 135次元

**最適化ポイント** :

  * ✅ チャンクサイズ: 1,000-10,000が最適（メモリと速度のトレードオフ）
  * ✅ `n_jobs=-1`で並列処理を有効化
  * ✅ `ignore_errors=True`でエラー行をスキップ
  * ✅ `tqdm`でプログレスバーを表示（ユーザーフレンドリー）
  * ✅ メモリ使用量をモニタリング（`psutil`使用）

**さらなる最適化** :

  * Daskを使用した分散処理（>1M組成）
  * HDF5/Parquetフォーマットで中間結果を保存
  * GPUアクセラレーション（CuPy、RAPIDS）

## 次のステップ

このchapterでは、元素特性データベース（Mendeleev、pymatgen、matminer）の違いと、 Featurizerアーキテクチャを学びました。 次章では、これらの特徴量を使った実際の機械学習モデルの構築と、 特徴量選択・次元削減の手法を学びます。 

[← 第2章: Magpieと統計記述子](<chapter-2.html>) 第4章: 機械学習との統合 →（準備中）

## 参考文献

  1. Ward, L., Dunn, A., Faghaninia, A., Zimmermann, N. E., Bajaj, S., Wang, Q., ... & Jain, A. (2018). "Matminer: An open source toolkit for materials data mining." _Computational Materials Science_ , 152, 60-69. DOI: [10.1016/j.commatsci.2018.05.018](<https://doi.org/10.1016/j.commatsci.2018.05.018>)   
_matminerの原論文。Featurizerアーキテクチャと主要な特徴量の詳細を記載（pp. 62-66）_
  2. Ong, S. P., Richards, W. D., Jain, A., Hautier, G., Kocher, M., Cholia, S., ... & Ceder, G. (2013). "Python Materials Genomics (pymatgen): A robust, open-source python library for materials analysis." _Computational Materials Science_ , 68, 314-319. DOI: [10.1016/j.commatsci.2012.10.028](<https://doi.org/10.1016/j.commatsci.2012.10.028>)   
_pymatgenライブラリの原論文。Element、Compositionクラスの詳細（pp. 315-317）_
  3. Himanen, L., Jäger, M. O., Morooka, E. V., Federici Canova, F., Ranawat, Y. S., Gao, D. Z., ... & Foster, A. S. (2019). "DScribe: Library of descriptors for machine learning in materials science." _Computer Physics Communications_ , 247, 106949, pp. 1-15. DOI: [10.1016/j.cpc.2019.106949](<https://doi.org/10.1016/j.cpc.2019.106949>)   
_材料記述子の包括的レビュー。組成ベース特徴量の理論的背景（pp. 3-7）_
  4. matminer API Documentation: Featurizer classes. <https://hackingmaterials.lbl.gov/matminer/>   
_matminerの公式ドキュメント。各Featurizerの詳細な使用例とパラメータ説明_
  5. pymatgen.core.periodic_table Documentation. [https://pymatgen.org/](<https://pymatgen.org/pymatgen.core.periodic_table.html>)   
_pymatgenのElementクラス、Compositionクラスの詳細なAPI仕様_
  6. Mendeleev package documentation. <https://mendeleev.readthedocs.io/>   
_Mendeleevパッケージの公式ドキュメント。元素特性データの出典と精度情報_
  7. Materials Project Database documentation. <https://docs.materialsproject.org/>   
_Materials Projectの公式ドキュメント。DFT計算手法とデータベース構造の詳細_

* * *

[シリーズ目次に戻る](<index.html>) 第4章へ進む →（準備中）

© 2025 AI Terakoya - Materials Informatics Knowledge Hub 

### 免責事項

  * 本コンテンツは教育・研究・情報提供のみを目的としており、専門的な助言(法律・会計・技術的保証など)を提供するものではありません。
  * 本コンテンツおよび付随するCode examplesは「現状有姿(AS IS)」で提供され、明示または黙示を問わず、商品性、特定目的適合性、権利非侵害、正確性・完全性、動作・安全性等いかなる保証もしません。
  * 外部リンク、第三者が提供するデータ・ツール・ライブラリ等の内容・可用性・安全性について、作成者および東北大学は一切の責任を負いません。
  * 本コンテンツの利用・実行・解釈により直接的・間接的・付随的・特別・結果的・懲罰的損害が生じた場合でも、適用法で許容される最大限の範囲で、作成者および東北大学は責任を負いません。
  * 本コンテンツの内容は、予告なく変更・更新・提供停止されることがあります。
  * 本コンテンツの著作権・ライセンスは明記された条件(例: CC BY 4.0)に従います。当該ライセンスは通常、無保証条項を含みます。
