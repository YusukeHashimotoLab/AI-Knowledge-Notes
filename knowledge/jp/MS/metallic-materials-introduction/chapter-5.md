---
title: "Chapter 5: Python実践：金属材料データ解析"
chapter_title: "Chapter 5: Python実践：金属材料データ解析"
subtitle: "Python Practice: Metallic Materials Data Analysis"
---

🌐 JP | [🇬🇧 EN](<../../../en/MS/metallic-materials-introduction/chapter-5.html>) | Last sync: 2025-11-16

# Chapter 5: Python実践：金属材料データ解析

Python Practice: Metallic Materials Data Analysis

[MS Dojo Top](<../index.html>) > [金属材料入門](<index.html>) > Chapter 5 

## 概要

本章では、Pythonを用いた金属材料データ解析の実践的な手法を学びます。Materials Project APIによる材料データベースへのアクセス、pymatgen/ASEを用いた構造解析、pycalphadによる状態図計算、そして機械学習による材料特性予測まで、現代の材料科学研究で必須のデータ駆動型アプローチを習得します。

### 本章の学習目標

  * Materials Project APIを用いて材料データベースから情報を取得できる
  * pymatgen/ASEで結晶構造を操作・可視化・解析できる
  * pycalphadで多元系状態図を計算し、相安定性を評価できる
  * pandas/numpyで大規模材料データを処理・統計解析できる
  * 機械学習（scikit-learn）で材料特性を予測するモデルを構築できる
  * 高スループット計算ワークフローを設計・実装できる
  * データ可視化により材料トレンドを発見できる

**⚠️ 環境準備** : 本章のコードを実行するには、以下のライブラリが必要です。  
`pip install pymatgen mp-api ase pycalphad scikit-learn pandas matplotlib seaborn plotly`  
Materials Project APIキーは[materialsproject.org](<https://materialsproject.org/>)で無料取得できます。 

## 5.1 Materials Project APIによるデータ取得

### 5.1.1 Materials Projectとは

Materials Project（MP）は、米国Lawrence Berkeley National Laboratoryが運営する材料データベースで、150,000種類以上の材料の第一原理計算データを提供しています。結晶構造、バンドギャップ、弾性定数、形成エネルギー、状態図などの情報にPython APIでアクセスできます。

### 5.1.2 API認証とクライアント初期化

Materials Project APIを使用するには、APIキーが必要です。ウェブサイトで無料登録後、`MPRester`クライアントを初期化します。

**Code Example 1: Materials Project APIの初期化と材料検索**
    
    
    from mp_api.client import MPRester
    import pandas as pd
    
    # APIキーを設定（実際のキーに置き換えてください）
    API_KEY = "your_api_key_here"
    
    def search_materials_by_formula(formula, api_key=API_KEY):
        """
        化学式で材料を検索し、基本情報を取得
    
        Parameters:
        -----------
        formula : str
            化学式（例: "Fe", "Al2O3", "Fe-Ni"）
        api_key : str
            Materials Project APIキー
    
        Returns:
        --------
        df : DataFrame
            材料情報のデータフレーム
        """
        with MPRester(api_key) as mpr:
            # 化学式で材料を検索
            docs = mpr.materials.summary.search(
                formula=formula,
                fields=["material_id", "formula_pretty", "structure",
                        "energy_per_atom", "formation_energy_per_atom",
                        "band_gap", "is_stable", "e_above_hull"]
            )
    
            if not docs:
                print(f"No materials found for formula: {formula}")
                return pd.DataFrame()
    
            # データフレームに変換
            data = []
            for doc in docs:
                data.append({
                    'Material ID': doc.material_id,
                    'Formula': doc.formula_pretty,
                    'Energy [eV/atom]': doc.energy_per_atom,
                    'Formation Energy [eV/atom]': doc.formation_energy_per_atom,
                    'Band Gap [eV]': doc.band_gap,
                    'Stable': doc.is_stable,
                    'E above hull [eV/atom]': doc.e_above_hull
                })
    
            df = pd.DataFrame(data)
            return df
    
    # 使用例: 鉄（Fe）の材料を検索
    print("=== Searching for Fe materials ===")
    fe_materials = search_materials_by_formula("Fe")
    print(fe_materials.head(10))
    
    # フィルタリング: 安定な材料のみ
    stable_fe = fe_materials[fe_materials['Stable'] == True]
    print(f"\nNumber of stable Fe materials: {len(stable_fe)}")
    
    # Fe-Ni合金の検索
    print("\n=== Searching for Fe-Ni alloys ===")
    fe_ni_alloys = search_materials_by_formula("Fe-Ni")
    print(fe_ni_alloys.head(10))
    
    # 統計情報
    print("\n=== Statistics for Fe-Ni alloys ===")
    print(f"Total materials: {len(fe_ni_alloys)}")
    print(f"Average formation energy: {fe_ni_alloys['Formation Energy [eV/atom]'].mean():.3f} eV/atom")
    print(f"Stable materials: {fe_ni_alloys['Stable'].sum()}")
    

### 5.1.3 構造情報の取得と可視化

Materials Projectから取得した構造データは、pymatgenのStructureオブジェクトとして提供されます。これを用いて結晶構造を解析・可視化できます。

**Code Example 2: 構造データの取得と解析**
    
    
    from mp_api.client import MPRester
    from pymatgen.core import Structure
    import matplotlib.pyplot as plt
    
    def get_structure_analysis(material_id, api_key=API_KEY):
        """
        Material IDから構造を取得し、詳細解析
    
        Parameters:
        -----------
        material_id : str
            Materials Project ID（例: "mp-13"）
        api_key : str
            APIキー
    
        Returns:
        --------
        structure : Structure
            pymatgen Structureオブジェクト
        """
        with MPRester(api_key) as mpr:
            # 構造データを取得
            structure = mpr.get_structure_by_material_id(material_id)
    
            print(f"=== Structure Analysis for {material_id} ===")
            print(f"Formula: {structure.composition.reduced_formula}")
            print(f"Space Group: {structure.get_space_group_info()}")
            print(f"Lattice Parameters:")
            print(f"  a = {structure.lattice.a:.4f} Å")
            print(f"  b = {structure.lattice.b:.4f} Å")
            print(f"  c = {structure.lattice.c:.4f} Å")
            print(f"  α = {structure.lattice.alpha:.2f}°")
            print(f"  β = {structure.lattice.beta:.2f}°")
            print(f"  γ = {structure.lattice.gamma:.2f}°")
            print(f"Volume: {structure.volume:.4f} Å³")
            print(f"Density: {structure.density:.4f} g/cm³")
    
            # 原子数と組成
            print(f"\nComposition:")
            for elem, amount in structure.composition.items():
                print(f"  {elem}: {amount:.2f}")
    
            # 最近接距離
            print(f"\nNearest neighbor distances:")
            neighbors = structure.get_all_neighbors(r=3.5)
            for i, site_neighbors in enumerate(neighbors[:3]):  # 最初の3サイト
                if site_neighbors:
                    min_dist = min([n.nn_distance for n in site_neighbors])
                    print(f"  Site {i} ({structure[i].species_string}): {min_dist:.3f} Å")
    
            return structure
    
    # 使用例: BCC鉄（mp-13）の構造解析
    bcc_fe_structure = get_structure_analysis("mp-13")
    
    # 複数材料の密度比較
    material_ids = ["mp-13", "mp-79", "mp-134"]  # Fe (BCC), Ni (FCC), Al (FCC)
    densities = []
    formulas = []
    
    with MPRester(API_KEY) as mpr:
        for mat_id in material_ids:
            structure = mpr.get_structure_by_material_id(mat_id)
            formulas.append(structure.composition.reduced_formula)
            densities.append(structure.density)
    
    # 可視化
    plt.figure(figsize=(10, 6))
    plt.bar(formulas, densities, color=['steelblue', 'lightcoral', 'lightgreen'])
    plt.ylabel('Density [g/cm³]', fontsize=12)
    plt.title('Density Comparison of Metallic Elements', fontsize=14)
    plt.grid(axis='y', alpha=0.3)
    for i, (formula, density) in enumerate(zip(formulas, densities)):
        plt.text(i, density + 0.2, f'{density:.2f}', ha='center', fontsize=11)
    plt.tight_layout()
    plt.savefig('density_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    

### 5.1.4 バッチ処理：複数材料の一括データ取得

高スループット解析では、多数の材料データを一括取得し、統計解析や機械学習に使用します。

**Code Example 3: 遷移金属材料の一括取得と統計解析**
    
    
    from mp_api.client import MPRester
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    def batch_download_transition_metals(api_key=API_KEY):
        """
        遷移金属元素の材料データを一括ダウンロード
    
        Returns:
        --------
        df : DataFrame
            材料データベース
        """
        transition_metals = ['Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn']
    
        all_data = []
    
        with MPRester(api_key) as mpr:
            for element in transition_metals:
                print(f"Downloading data for {element}...")
    
                # 元素の純物質を検索
                docs = mpr.materials.summary.search(
                    chemsys=element,
                    fields=["material_id", "formula_pretty", "formation_energy_per_atom",
                            "energy_per_atom", "band_gap", "density",
                            "e_above_hull", "is_stable"]
                )
    
                for doc in docs:
                    all_data.append({
                        'Material ID': doc.material_id,
                        'Element': element,
                        'Formula': doc.formula_pretty,
                        'Formation Energy [eV/atom]': doc.formation_energy_per_atom,
                        'Energy [eV/atom]': doc.energy_per_atom,
                        'Band Gap [eV]': doc.band_gap,
                        'Density [g/cm³]': doc.density,
                        'E above hull [eV/atom]': doc.e_above_hull,
                        'Stable': doc.is_stable
                    })
    
        df = pd.DataFrame(all_data)
        return df
    
    # データダウンロード
    print("=== Batch Download Transition Metal Materials ===")
    tm_data = batch_download_transition_metals()
    
    # 統計解析
    print(f"\nTotal materials: {len(tm_data)}")
    print(f"Stable materials: {tm_data['Stable'].sum()}")
    
    # 元素ごとの統計
    print("\n=== Statistics by Element ===")
    element_stats = tm_data.groupby('Element').agg({
        'Formation Energy [eV/atom]': 'mean',
        'Density [g/cm³]': 'mean',
        'Stable': 'sum'
    }).round(3)
    print(element_stats)
    
    # 可視化: 形成エネルギーの分布
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. 形成エネルギー vs 密度
    stable_data = tm_data[tm_data['Stable'] == True]
    axes[0, 0].scatter(stable_data['Density [g/cm³]'],
                        stable_data['Formation Energy [eV/atom]'],
                        c=stable_data['Band Gap [eV]'], cmap='viridis',
                        s=50, alpha=0.6)
    axes[0, 0].set_xlabel('Density [g/cm³]', fontsize=11)
    axes[0, 0].set_ylabel('Formation Energy [eV/atom]', fontsize=11)
    axes[0, 0].set_title('Formation Energy vs. Density', fontsize=12)
    cbar = plt.colorbar(axes[0, 0].collections[0], ax=axes[0, 0])
    cbar.set_label('Band Gap [eV]', fontsize=10)
    axes[0, 0].grid(alpha=0.3)
    
    # 2. 元素ごとの形成エネルギー分布
    sns.boxplot(data=tm_data, x='Element', y='Formation Energy [eV/atom]',
                ax=axes[0, 1], palette='Set2')
    axes[0, 1].set_title('Formation Energy Distribution by Element', fontsize=12)
    axes[0, 1].set_ylabel('Formation Energy [eV/atom]', fontsize=11)
    axes[0, 1].tick_params(axis='x', rotation=45)
    axes[0, 1].grid(axis='y', alpha=0.3)
    
    # 3. 密度分布
    axes[1, 0].hist(stable_data['Density [g/cm³]'], bins=20,
                    color='steelblue', alpha=0.7, edgecolor='black')
    axes[1, 0].set_xlabel('Density [g/cm³]', fontsize=11)
    axes[1, 0].set_ylabel('Count', fontsize=11)
    axes[1, 0].set_title('Density Distribution (Stable Materials)', fontsize=12)
    axes[1, 0].grid(alpha=0.3)
    
    # 4. 安定性の比率
    stability_counts = tm_data.groupby('Element')['Stable'].value_counts().unstack(fill_value=0)
    stability_counts.plot(kind='bar', stacked=True, ax=axes[1, 1],
                           color=['lightcoral', 'lightgreen'],
                           legend=True)
    axes[1, 1].set_title('Stability Count by Element', fontsize=12)
    axes[1, 1].set_ylabel('Count', fontsize=11)
    axes[1, 1].tick_params(axis='x', rotation=45)
    axes[1, 1].legend(['Unstable', 'Stable'], fontsize=10)
    
    plt.tight_layout()
    plt.savefig('transition_metal_statistics.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # データをCSVに保存
    tm_data.to_csv('transition_metals_data.csv', index=False)
    print("\nData saved to 'transition_metals_data.csv'")
    

## 5.2 pymatgenによる構造解析

### 5.2.1 結晶構造の生成と操作

pymatgenは、結晶構造の生成、変形、対称性解析などの高度な機能を提供します。

**Code Example 4: 結晶構造の生成とスーパーセル作成**
    
    
    from pymatgen.core import Structure, Lattice
    import numpy as np
    from pymatgen.io.ase import AseAtomsAdaptor
    from ase.visualize import view
    
    def create_fcc_structure(element, lattice_constant):
        """
        FCC構造を生成
    
        Parameters:
        -----------
        element : str
            元素記号
        lattice_constant : float
            格子定数 [Å]
    
        Returns:
        --------
        structure : Structure
            FCC構造
        """
        lattice = Lattice.cubic(lattice_constant)
        coords = [[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]]
        structure = Structure(lattice, [element]*4, coords)
        return structure
    
    # FCC Niの生成
    fcc_ni = create_fcc_structure("Ni", 3.52)
    print("=== FCC Ni Structure ===")
    print(fcc_ni)
    
    # スーパーセルの作成（2x2x2）
    supercell = fcc_ni.copy()
    supercell.make_supercell([2, 2, 2])
    print(f"\n=== Supercell (2x2x2) ===")
    print(f"Number of atoms: {len(supercell)}")
    print(f"Volume: {supercell.volume:.2f} Å³")
    
    # 置換型固溶体の作成（Fe-Ni）
    def create_substitutional_alloy(base_structure, substitute_element, substitute_fraction):
        """
        置換型固溶体を作成
    
        Parameters:
        -----------
        base_structure : Structure
            ベース構造
        substitute_element : str
            置換元素
        substitute_fraction : float
            置換割合（0-1）
    
        Returns:
        --------
        alloy_structure : Structure
            合金構造
        """
        alloy = base_structure.copy()
        n_substitute = int(len(alloy) * substitute_fraction)
    
        # ランダムにサイトを選択して置換
        np.random.seed(42)  # 再現性のため
        substitute_indices = np.random.choice(len(alloy), n_substitute, replace=False)
    
        for idx in substitute_indices:
            alloy[idx] = substitute_element
    
        return alloy
    
    # Fe0.5Ni0.5合金の作成（スーパーセルから）
    fe_ni_alloy = create_substitutional_alloy(supercell, "Fe", 0.5)
    print(f"\n=== Fe-Ni Alloy (50% substitution) ===")
    print(f"Composition: {fe_ni_alloy.composition}")
    
    # 最近接距離分析
    from pymatgen.analysis.structure_analyzer import VoronoiConnectivity
    
    def analyze_nearest_neighbors(structure):
        """
        最近接原子間距離を解析
        """
        print(f"\n=== Nearest Neighbor Analysis ===")
        all_neighbors = structure.get_all_neighbors(r=4.0)
    
        distances = []
        for site_neighbors in all_neighbors:
            if site_neighbors:
                min_dist = min([n.nn_distance for n in site_neighbors])
                distances.append(min_dist)
    
        distances = np.array(distances)
        print(f"Mean nearest neighbor distance: {distances.mean():.3f} Å")
        print(f"Std deviation: {distances.std():.3f} Å")
        print(f"Min distance: {distances.min():.3f} Å")
        print(f"Max distance: {distances.max():.3f} Å")
    
        return distances
    
    nn_distances = analyze_nearest_neighbors(fe_ni_alloy)
    
    # 配位数の計算
    def calculate_coordination_numbers(structure, cutoff=3.5):
        """
        配位数を計算
        """
        all_neighbors = structure.get_all_neighbors(r=cutoff)
        coordination_numbers = [len(neighbors) for neighbors in all_neighbors]
    
        print(f"\n=== Coordination Number Analysis (cutoff={cutoff} Å) ===")
        print(f"Mean coordination number: {np.mean(coordination_numbers):.2f}")
        print(f"Most common: {np.bincount(coordination_numbers).argmax()}")
    
        return coordination_numbers
    
    coord_nums = calculate_coordination_numbers(fe_ni_alloy)
    
    # ASEへの変換と可視化（オプション）
    adaptor = AseAtomsAdaptor()
    ase_atoms = adaptor.get_atoms(fe_ni_alloy)
    # view(ase_atoms)  # ASE GUIで可視化（インタラクティブ環境）
    

### 5.2.2 対称性解析と空間群決定

pymatgenは、空間群の自動判定や対称性操作の取得が可能です。

**Code Example 5: 対称性解析**
    
    
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
    from pymatgen.core import Structure
    
    def symmetry_analysis(structure, symprec=0.1):
        """
        結晶構造の対称性を詳細解析
    
        Parameters:
        -----------
        structure : Structure
            解析対象の構造
        symprec : float
            対称性判定の精度 [Å]
    
        Returns:
        --------
        analysis_dict : dict
            解析結果
        """
        sga = SpacegroupAnalyzer(structure, symprec=symprec)
    
        print(f"=== Symmetry Analysis ===")
        print(f"Space Group: {sga.get_space_group_symbol()}")
        print(f"Space Group Number: {sga.get_space_group_number()}")
        print(f"Point Group: {sga.get_point_group_symbol()}")
        print(f"Crystal System: {sga.get_crystal_system()}")
        print(f"Lattice Type: {sga.get_lattice_type()}")
    
        # 対称性操作の数
        symm_ops = sga.get_symmetry_operations()
        print(f"Number of symmetry operations: {len(symm_ops)}")
    
        # プリミティブセルへの変換
        primitive = sga.get_primitive_standard_structure()
        print(f"\nPrimitive cell:")
        print(f"  Number of atoms: {len(primitive)}")
        print(f"  Volume: {primitive.volume:.4f} Å³")
    
        # コンベンショナルセルへの変換
        conventional = sga.get_conventional_standard_structure()
        print(f"\nConventional cell:")
        print(f"  Number of atoms: {len(conventional)}")
        print(f"  Volume: {conventional.volume:.4f} Å³")
    
        return {
            'space_group': sga.get_space_group_symbol(),
            'sg_number': sga.get_space_group_number(),
            'point_group': sga.get_point_group_symbol(),
            'crystal_system': sga.get_crystal_system(),
            'primitive': primitive,
            'conventional': conventional
        }
    
    # FCC Ni構造の対称性解析
    fcc_ni_structure = create_fcc_structure("Ni", 3.52)
    ni_symmetry = symmetry_analysis(fcc_ni_structure)
    
    # BCC Fe構造の生成と解析
    def create_bcc_structure(element, lattice_constant):
        """BCC構造を生成"""
        lattice = Lattice.cubic(lattice_constant)
        coords = [[0, 0, 0], [0.5, 0.5, 0.5]]
        structure = Structure(lattice, [element]*2, coords)
        return structure
    
    bcc_fe_structure = create_bcc_structure("Fe", 2.87)
    print("\n" + "="*50)
    fe_symmetry = symmetry_analysis(bcc_fe_structure)
    
    # 比較表
    import pandas as pd
    comparison_df = pd.DataFrame({
        'Property': ['Space Group', 'SG Number', 'Point Group', 'Crystal System',
                     'Atoms in primitive', 'Atoms in conventional'],
        'FCC Ni': [ni_symmetry['space_group'], ni_symmetry['sg_number'],
                   ni_symmetry['point_group'], ni_symmetry['crystal_system'],
                   len(ni_symmetry['primitive']), len(ni_symmetry['conventional'])],
        'BCC Fe': [fe_symmetry['space_group'], fe_symmetry['sg_number'],
                   fe_symmetry['point_group'], fe_symmetry['crystal_system'],
                   len(fe_symmetry['primitive']), len(fe_symmetry['conventional'])]
    })
    print("\n=== FCC vs BCC Comparison ===")
    print(comparison_df.to_string(index=False))
    

## 5.3 pycalphadによる状態図計算

### 5.3.1 CALPHADアプローチとpycalphad

CALPHAD (CALculation of PHAse Diagrams) 法は、熱力学データベースを用いて多元系状態図を計算する手法です。pycalphadはPythonでCALPHAD計算を実行するライブラリで、TDB（Thermodynamic DataBase）ファイルを読み込み、平衡計算が可能です。

**Code Example 6: Fe-C二元系状態図の計算**
    
    
    from pycalphad import Database, equilibrium, variables as v
    import matplotlib.pyplot as plt
    import numpy as np
    
    def calculate_binary_phase_diagram(tdb_file, elements, x_component, T_range, x_range):
        """
        二元系状態図を計算
    
        Parameters:
        -----------
        tdb_file : str
            TDBファイルパス
        elements : list
            元素リスト（例: ['FE', 'C']）
        x_component : str
            横軸の成分（例: 'C'）
        T_range : tuple
            温度範囲 [K] (T_min, T_max, num_points)
        x_range : tuple
            組成範囲 (x_min, x_max, num_points)
    
        Returns:
        --------
        eq : xarray.Dataset
            平衡計算結果
        """
        # データベース読み込み
        db = Database(tdb_file)
    
        # 温度・組成範囲の設定
        temps = np.linspace(T_range[0], T_range[1], T_range[2])
        compositions = np.linspace(x_range[0], x_range[1], x_range[2])
    
        # 平衡計算
        eq = equilibrium(db, elements, 'BCC_A2',
                          {v.X(x_component): compositions, v.T: temps, v.P: 101325},
                          calc_opts={'pbar': False})
    
        return eq, temps, compositions
    
    # Fe-C系状態図計算の例（実際にはTDBファイルが必要）
    # 注: 実際の実行にはFe-CのTDBファイルが必要です
    # ここでは疑似コードとして概念を示します
    
    def plot_fe_c_phase_diagram_conceptual():
        """
        Fe-C系状態図の概念的プロット（実データベースなし）
        """
        # 温度と炭素濃度の範囲
        T = np.linspace(500, 1600, 100)  # K to °C変換: T - 273
        C_wt = np.linspace(0, 6.7, 100)  # wt% C
    
        fig, ax = plt.subplots(figsize=(12, 8))
    
        # A1線（727°C、共析温度）
        A1_temp = 727
        ax.axhline(y=A1_temp, color='blue', linestyle='--', linewidth=2,
                    label='A1 (Eutectoid, 727°C)')
    
        # A3線（Ferrite-Austenite境界、概念的）
        A3_C = np.linspace(0, 0.8, 50)
        A3_T = 912 - 200 * A3_C
        ax.plot(A3_C, A3_T, 'r-', linewidth=2, label='A3 (Ferrite-Austenite)')
    
        # Acm線（Austenite-Cementite境界、概念的）
        Acm_C = np.linspace(0.8, 6.7, 50)
        Acm_T = 727 + (1147 - 727) * (Acm_C - 0.8) / (6.7 - 0.8)
        ax.plot(Acm_C, Acm_T, 'g-', linewidth=2, label='Acm (Austenite-Cementite)')
    
        # 共析点
        ax.plot(0.8, 727, 'ko', markersize=12, label='Eutectoid Point (0.8%C, 727°C)')
    
        # 共晶点（4.3%C、1147°C）
        ax.plot(4.3, 1147, 'rs', markersize=12, label='Eutectic Point (4.3%C, 1147°C)')
    
        # 相領域のラベル
        ax.text(0.2, 500, 'Ferrite (α)', fontsize=14, fontweight='bold')
        ax.text(0.2, 850, 'Austenite (γ)', fontsize=14, fontweight='bold')
        ax.text(1.5, 650, 'Ferrite + Pearlite', fontsize=12)
        ax.text(5.5, 900, 'Austenite + Cementite', fontsize=12)
    
        ax.set_xlabel('Carbon Content [wt%]', fontsize=13)
        ax.set_ylabel('Temperature [°C]', fontsize=13)
        ax.set_title('Fe-C Phase Diagram (Conceptual)', fontsize=15, fontweight='bold')
        ax.legend(fontsize=11, loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 6.7)
        ax.set_ylim(400, 1600)
    
        plt.tight_layout()
        plt.savefig('fe_c_phase_diagram_conceptual.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    plot_fe_c_phase_diagram_conceptual()
    
    print("=== Fe-C Phase Diagram Key Points ===")
    print("Eutectoid: 0.8 wt% C, 727°C (Austenite → Ferrite + Cementite)")
    print("Eutectic: 4.3 wt% C, 1147°C (Liquid → Austenite + Cementite)")
    print("A1 temperature: 727°C (Ferrite-Austenite-Cementite equilibrium)")
    print("A3 temperature: Variable (Ferrite-Austenite boundary, composition-dependent)")
    

**ℹ️ TDBファイルについて** : pycalphadで実際の状態図計算を行うには、TDB（Thermodynamic DataBase）ファイルが必要です。Fe-C系のTDBファイルは、[pycalphad公式リポジトリ](<https://github.com/pycalphad/pycalphad>)のexamplesや、[Thermocalc](<https://www.thermocalc.com/>)などで入手できます。 

## 5.4 機械学習による材料特性予測

### 5.4.1 材料記述子（Material Descriptors）

機械学習モデルに材料を入力するには、化学組成や構造を数値ベクトル（記述子）に変換する必要があります。一般的な記述子には、元素の電気陰性度、原子半径、価電子数などの平均値や、結晶構造の対称性指標があります。

**Code Example 7: 材料記述子の生成と機械学習モデル構築**
    
    
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import Ridge
    from sklearn.metrics import mean_absolute_error, r2_score
    import matplotlib.pyplot as plt
    
    # 疑似データセット作成（実際にはMaterials Projectからダウンロード）
    def create_sample_dataset(n_samples=200):
        """
        金属材料の疑似データセットを作成
    
        Returns:
        --------
        df : DataFrame
            材料特性データ
        """
        np.random.seed(42)
    
        # 記述子（特徴量）
        atomic_radius = np.random.uniform(1.0, 2.0, n_samples)  # Å
        electronegativity = np.random.uniform(1.5, 2.5, n_samples)
        valence_electrons = np.random.randint(3, 11, n_samples)
        density = np.random.uniform(2.0, 10.0, n_samples)  # g/cm³
        formation_energy = np.random.uniform(-2.0, 0.5, n_samples)  # eV/atom
    
        # ターゲット: 弾性率（疑似的に生成）
        # 実際のトレンド: 密度が高く、電気陰性度が高いほど硬い
        elastic_modulus = (50 + 30 * density + 20 * electronegativity +
                            5 * valence_electrons - 10 * formation_energy +
                            np.random.normal(0, 20, n_samples))  # GPa
    
        df = pd.DataFrame({
            'Atomic Radius [Å]': atomic_radius,
            'Electronegativity': electronegativity,
            'Valence Electrons': valence_electrons,
            'Density [g/cm³]': density,
            'Formation Energy [eV/atom]': formation_energy,
            'Elastic Modulus [GPa]': elastic_modulus
        })
    
        return df
    
    # データセット作成
    materials_df = create_sample_dataset(200)
    print("=== Material Dataset ===")
    print(materials_df.head(10))
    print(f"\nDataset shape: {materials_df.shape}")
    print(f"Statistical summary:")
    print(materials_df.describe())
    
    # 特徴量とターゲットの分離
    X = materials_df.drop('Elastic Modulus [GPa]', axis=1)
    y = materials_df['Elastic Modulus [GPa]']
    
    # Train-Test分割
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                           random_state=42)
    
    print(f"\nTraining set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # 複数モデルの訓練と評価
    models = {
        'Ridge Regression': Ridge(alpha=1.0),
        'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=10,
                                                random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, max_depth=5,
                                                         learning_rate=0.1, random_state=42)
    }
    
    results = {}
    
    print("\n=== Model Training and Evaluation ===")
    for name, model in models.items():
        # 訓練
        model.fit(X_train, y_train)
    
        # 予測
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
    
        # 評価指標
        train_mae = mean_absolute_error(y_train, y_train_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
    
        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5,
                                      scoring='r2')
    
        results[name] = {
            'Train MAE': train_mae,
            'Test MAE': test_mae,
            'Train R²': train_r2,
            'Test R²': test_r2,
            'CV R² Mean': cv_scores.mean(),
            'CV R² Std': cv_scores.std(),
            'Predictions': y_test_pred
        }
    
        print(f"\n{name}:")
        print(f"  Train MAE: {train_mae:.2f} GPa, R² = {train_r2:.3f}")
        print(f"  Test MAE: {test_mae:.2f} GPa, R² = {test_r2:.3f}")
        print(f"  Cross-validation R²: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    
    # 最良モデルの選択
    best_model_name = max(results, key=lambda k: results[k]['Test R²'])
    print(f"\n=== Best Model: {best_model_name} ===")
    print(f"Test R² = {results[best_model_name]['Test R²']:.3f}")
    
    # 可視化: 予測 vs 実測
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    for i, (name, model) in enumerate(models.items()):
        y_pred = results[name]['Predictions']
    
        axes[i].scatter(y_test, y_pred, alpha=0.6, s=50, color='steelblue')
        axes[i].plot([y_test.min(), y_test.max()],
                      [y_test.min(), y_test.max()],
                      'r--', linewidth=2, label='Perfect Prediction')
    
        axes[i].set_xlabel('True Elastic Modulus [GPa]', fontsize=11)
        axes[i].set_ylabel('Predicted Elastic Modulus [GPa]', fontsize=11)
        axes[i].set_title(f'{name}\nR² = {results[name]["Test R²"]:.3f}', fontsize=12)
        axes[i].legend(fontsize=10)
        axes[i].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('ml_prediction_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 特徴量重要度（Random Forestの場合）
    if 'Random Forest' in models:
        rf_model = models['Random Forest']
        feature_importance = pd.DataFrame({
            'Feature': X.columns,
            'Importance': rf_model.feature_importances_
        }).sort_values('Importance', ascending=False)
    
        print("\n=== Feature Importance (Random Forest) ===")
        print(feature_importance.to_string(index=False))
    
        # 可視化
        plt.figure(figsize=(10, 6))
        plt.barh(feature_importance['Feature'], feature_importance['Importance'],
                  color='steelblue')
        plt.xlabel('Importance', fontsize=12)
        plt.title('Feature Importance for Elastic Modulus Prediction', fontsize=14)
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()
    

### 5.4.2 高度な記述子：Magpie特徴量

Magpie（Materials Agnostic Platform for Informatics and Exploration）は、化学組成から自動的に100種類以上の記述子を生成するツールです。pymatgenと統合されており、高精度な材料特性予測が可能です。

## 5.5 高スループット計算ワークフロー

### 5.5.1 ワークフロー設計

大規模な材料スクリーニングでは、データ取得→前処理→解析→予測のワークフローを自動化します。

**Code Example 8: 材料スクリーニングワークフロー**
    
    
    import pandas as pd
    import numpy as np
    from mp_api.client import MPRester
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import StandardScaler
    import matplotlib.pyplot as plt
    
    class MaterialScreeningWorkflow:
        """
        材料スクリーニング自動化ワークフロー
        """
    
        def __init__(self, api_key):
            self.api_key = api_key
            self.data = None
            self.model = None
            self.scaler = StandardScaler()
    
        def step1_data_collection(self, elements, max_materials=100):
            """
            Step 1: データ収集
            """
            print("=== Step 1: Data Collection ===")
            all_data = []
    
            with MPRester(self.api_key) as mpr:
                for element in elements:
                    print(f"Fetching data for {element}...")
                    docs = mpr.materials.summary.search(
                        chemsys=element,
                        fields=["material_id", "formula_pretty", "formation_energy_per_atom",
                                "band_gap", "density", "e_above_hull", "is_stable"],
                        num_chunks=1
                    )
    
                    for doc in docs[:max_materials]:
                        all_data.append({
                            'Material ID': doc.material_id,
                            'Formula': doc.formula_pretty,
                            'Formation Energy [eV/atom]': doc.formation_energy_per_atom,
                            'Band Gap [eV]': doc.band_gap,
                            'Density [g/cm³]': doc.density,
                            'E above hull [eV/atom]': doc.e_above_hull,
                            'Stable': doc.is_stable
                        })
    
            self.data = pd.DataFrame(all_data)
            print(f"Collected {len(self.data)} materials")
            return self.data
    
        def step2_preprocessing(self):
            """
            Step 2: データ前処理
            """
            print("\n=== Step 2: Preprocessing ===")
    
            # 欠損値の除去
            initial_count = len(self.data)
            self.data = self.data.dropna()
            print(f"Removed {initial_count - len(self.data)} samples with missing values")
    
            # 安定材料のみフィルタ
            stable_data = self.data[self.data['Stable'] == True]
            print(f"Stable materials: {len(stable_data)} / {len(self.data)}")
            self.data = stable_data
    
            return self.data
    
        def step3_feature_engineering(self):
            """
            Step 3: 特徴量エンジニアリング
            """
            print("\n=== Step 3: Feature Engineering ===")
    
            # 新しい特徴量の生成（例: 相対的安定性指標）
            self.data['Stability Score'] = -self.data['E above hull [eV/atom]']
    
            # 密度を対数変換（スケール調整）
            self.data['Log Density'] = np.log10(self.data['Density [g/cm³]'])
    
            print("New features created:")
            print(f"  - Stability Score")
            print(f"  - Log Density")
    
            return self.data
    
        def step4_screening(self, target_property, criteria):
            """
            Step 4: スクリーニング
    
            Parameters:
            -----------
            target_property : str
                スクリーニング対象の特性
            criteria : dict
                スクリーニング条件（例: {'min': 5.0, 'max': 10.0}）
            """
            print(f"\n=== Step 4: Screening for {target_property} ===")
            print(f"Criteria: {criteria}")
    
            screened = self.data.copy()
    
            if 'min' in criteria:
                screened = screened[screened[target_property] >= criteria['min']]
                print(f"  - {target_property} >= {criteria['min']}: {len(screened)} materials")
    
            if 'max' in criteria:
                screened = screened[screened[target_property] <= criteria['max']]
                print(f"  - {target_property} <= {criteria['max']}: {len(screened)} materials")
    
            print(f"\nScreened candidates: {len(screened)}")
    
            return screened
    
        def step5_visualization(self, screened_data):
            """
            Step 5: 結果の可視化
            """
            print("\n=== Step 5: Visualization ===")
    
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
            # 1. 形成エネルギー vs 密度
            axes[0, 0].scatter(self.data['Density [g/cm³]'],
                                self.data['Formation Energy [eV/atom]'],
                                alpha=0.3, s=30, color='gray', label='All materials')
            axes[0, 0].scatter(screened_data['Density [g/cm³]'],
                                screened_data['Formation Energy [eV/atom]'],
                                alpha=0.7, s=50, color='red', label='Screened')
            axes[0, 0].set_xlabel('Density [g/cm³]', fontsize=11)
            axes[0, 0].set_ylabel('Formation Energy [eV/atom]', fontsize=11)
            axes[0, 0].set_title('Formation Energy vs. Density', fontsize=12)
            axes[0, 0].legend()
            axes[0, 0].grid(alpha=0.3)
    
            # 2. 密度分布
            axes[0, 1].hist(self.data['Density [g/cm³]'], bins=30, alpha=0.5,
                             color='gray', label='All', edgecolor='black')
            axes[0, 1].hist(screened_data['Density [g/cm³]'], bins=20, alpha=0.7,
                             color='red', label='Screened', edgecolor='black')
            axes[0, 1].set_xlabel('Density [g/cm³]', fontsize=11)
            axes[0, 1].set_ylabel('Count', fontsize=11)
            axes[0, 1].set_title('Density Distribution', fontsize=12)
            axes[0, 1].legend()
            axes[0, 1].grid(axis='y', alpha=0.3)
    
            # 3. E above hull分布
            axes[1, 0].hist(self.data['E above hull [eV/atom]'], bins=30, alpha=0.5,
                             color='gray', label='All', edgecolor='black')
            axes[1, 0].hist(screened_data['E above hull [eV/atom]'], bins=20, alpha=0.7,
                             color='red', label='Screened', edgecolor='black')
            axes[1, 0].set_xlabel('E above hull [eV/atom]', fontsize=11)
            axes[1, 0].set_ylabel('Count', fontsize=11)
            axes[1, 0].set_title('Stability Distribution', fontsize=12)
            axes[1, 0].legend()
            axes[1, 0].grid(axis='y', alpha=0.3)
    
            # 4. Top候補材料のリスト
            top_candidates = screened_data.nsmallest(10, 'Formation Energy [eV/atom]')
            axes[1, 1].axis('off')
            table_data = top_candidates[['Formula', 'Density [g/cm³]',
                                           'Formation Energy [eV/atom]']].values
            table = axes[1, 1].table(cellText=table_data,
                                      colLabels=['Formula', 'Density', 'Form. Energy'],
                                      cellLoc='center', loc='center',
                                      colWidths=[0.3, 0.3, 0.4])
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1, 2)
            axes[1, 1].set_title('Top 10 Candidates', fontsize=12, pad=20)
    
            plt.tight_layout()
            plt.savefig('screening_results.png', dpi=300, bbox_inches='tight')
            plt.show()
    
    # ワークフロー実行例（APIキーが必要）
    if __name__ == "__main__":
        # 初期化
        API_KEY = "your_api_key_here"  # 実際のキーに置き換え
        workflow = MaterialScreeningWorkflow(API_KEY)
    
        # Step 1: データ収集（遷移金属）
        elements = ['Fe', 'Ni', 'Cu', 'Ti', 'Al']
        # workflow.step1_data_collection(elements, max_materials=50)
    
        # Step 2: 前処理
        # workflow.step2_preprocessing()
    
        # Step 3: 特徴量エンジニアリング
        # workflow.step3_feature_engineering()
    
        # Step 4: スクリーニング（密度5-10 g/cm³の材料）
        # screened = workflow.step4_screening('Density [g/cm³]', {'min': 5.0, 'max': 10.0})
    
        # Step 5: 可視化
        # workflow.step5_visualization(screened)
    
        # 結果の保存
        # screened.to_csv('screened_materials.csv', index=False)
        # print("\nScreened materials saved to 'screened_materials.csv'")
    
        print("\n=== Workflow Conceptual Example ===")
        print("The workflow demonstrates automated material screening:")
        print("1. Collect data from Materials Project API")
        print("2. Preprocess and filter stable materials")
        print("3. Engineer new features")
        print("4. Screen by target properties (density, formation energy, etc.)")
        print("5. Visualize and export results")
    

## 演習問題

📝 Exercise 1: Materials Project APIでAl合金を検索 Easy

**問題** : Materials Project APIを用いて、Al-Cu系合金（Al、Cu、およびその組み合わせ）を検索し、安定な材料のみを抽出してください。形成エネルギーが最も低い上位5つの材料の化学式とMaterial IDを表示してください。

💡 解答例
    
    
    from mp_api.client import MPRester
    import pandas as pd
    
    API_KEY = "your_api_key_here"
    
    def search_al_cu_alloys(api_key=API_KEY):
        """Al-Cu系合金を検索"""
        with MPRester(api_key) as mpr:
            # Al-Cu化学系で検索
            docs = mpr.materials.summary.search(
                chemsys="Al-Cu",
                fields=["material_id", "formula_pretty", "formation_energy_per_atom",
                        "is_stable", "e_above_hull"]
            )
    
            data = []
            for doc in docs:
                data.append({
                    'Material ID': doc.material_id,
                    'Formula': doc.formula_pretty,
                    'Formation Energy [eV/atom]': doc.formation_energy_per_atom,
                    'Stable': doc.is_stable,
                    'E above hull [eV/atom]': doc.e_above_hull
                })
    
            df = pd.DataFrame(data)
            return df
    
    # 検索実行
    print("=== Searching Al-Cu Alloys ===")
    al_cu_alloys = search_al_cu_alloys()
    
    # 安定材料のみフィルタ
    stable_alloys = al_cu_alloys[al_cu_alloys['Stable'] == True]
    print(f"\nTotal materials: {len(al_cu_alloys)}")
    print(f"Stable materials: {len(stable_alloys)}")
    
    # 形成エネルギーが最も低い上位5つ
    top5 = stable_alloys.nsmallest(5, 'Formation Energy [eV/atom]')
    
    print("\n=== Top 5 Stable Al-Cu Alloys (Lowest Formation Energy) ===")
    print(top5[['Material ID', 'Formula', 'Formation Energy [eV/atom]']].to_string(index=False))
    
    # 結果の考察
    print("\n=== Analysis ===")
    print(f"Most stable alloy: {top5.iloc[0]['Formula']} ({top5.iloc[0]['Material ID']})")
    print(f"Formation energy: {top5.iloc[0]['Formation Energy [eV/atom]']:.3f} eV/atom")
    print("\nStable Al-Cu compounds typically include Al2Cu, AlCu, and Cu-rich phases.")
    print("Formation energy < 0 indicates thermodynamic stability relative to pure elements.")
    

**期待される出力** : Al2Cu、AlCu等の安定化合物のリストと形成エネルギー

**解説** : `chemsys="Al-Cu"`でAl-Cu二元系のすべての化合物を検索できます。`is_stable=True`でフィルタリングし、形成エネルギーが負の値（安定）の材料を抽出します。

📝 Exercise 2: pymatgenでスーパーセル作成と配位数計算 Medium

**問題** : pymatgenを用いてFCC Cu構造（格子定数3.61 Å）の3×3×3スーパーセルを作成し、全原子の配位数を計算してください。また、最近接原子間距離の平均と標準偏差を求めてください。

💡 解答例
    
    
    from pymatgen.core import Structure, Lattice
    import numpy as np
    
    def create_fcc_cu_supercell():
        """FCC Cu構造の3x3x3スーパーセルを作成"""
        # FCC単位格子
        lattice = Lattice.cubic(3.61)
        coords = [[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]]
        fcc_cu = Structure(lattice, ["Cu"]*4, coords)
    
        print("=== FCC Cu Unit Cell ===")
        print(f"Lattice parameter: 3.61 Å")
        print(f"Number of atoms: {len(fcc_cu)}")
    
        # スーパーセル作成
        supercell = fcc_cu.copy()
        supercell.make_supercell([3, 3, 3])
    
        print(f"\n=== 3x3x3 Supercell ===")
        print(f"Number of atoms: {len(supercell)}")
        print(f"Volume: {supercell.volume:.2f} Å³")
    
        return supercell
    
    # スーパーセル作成
    cu_supercell = create_fcc_cu_supercell()
    
    # 配位数計算（cutoff = 3.0 Å、最近接のみ）
    print("\n=== Coordination Number Analysis ===")
    cutoff = 3.0  # Å
    all_neighbors = cu_supercell.get_all_neighbors(r=cutoff)
    
    coordination_numbers = [len(neighbors) for neighbors in all_neighbors]
    
    print(f"Cutoff distance: {cutoff} Å")
    print(f"Coordination numbers:")
    print(f"  Mean: {np.mean(coordination_numbers):.2f}")
    print(f"  Std: {np.std(coordination_numbers):.3f}")
    print(f"  Min: {np.min(coordination_numbers)}")
    print(f"  Max: {np.max(coordination_numbers)}")
    
    # 最近接距離の統計
    print("\n=== Nearest Neighbor Distance Analysis ===")
    nn_distances = []
    
    for site_neighbors in all_neighbors:
        if site_neighbors:
            min_dist = min([n.nn_distance for n in site_neighbors])
            nn_distances.append(min_dist)
    
    nn_distances = np.array(nn_distances)
    
    print(f"Mean nearest neighbor distance: {nn_distances.mean():.4f} Å")
    print(f"Std deviation: {nn_distances.std():.4f} Å")
    print(f"Min distance: {nn_distances.min():.4f} Å")
    print(f"Max distance: {nn_distances.max():.4f} Å")
    
    # 理論値との比較
    theoretical_nn = 3.61 / np.sqrt(2)  # FCC最近接距離 = a/√2
    print(f"\nTheoretical nearest neighbor distance (a/√2): {theoretical_nn:.4f} Å")
    print(f"Difference from calculated: {abs(nn_distances.mean() - theoretical_nn):.4f} Å")
    
    # FCC構造の配位数は12
    print("\n=== Verification ===")
    if np.mean(coordination_numbers) == 12:
        print("✓ Coordination number = 12 (FCC structure confirmed)")
    else:
        print(f"⚠ Coordination number = {np.mean(coordination_numbers):.1f} (expected 12)")
    

**期待される出力** :
    
    
    Number of atoms: 108 (3×3×3×4)
    Mean coordination number: 12.00
    Mean nearest neighbor distance: 2.552 Å (= 3.61/√2)
    

**解説** : FCC構造の配位数は12です。最近接距離は格子定数\\(a\\)から\\(a/\sqrt{2}\\)で計算されます。スーパーセル内部の原子はすべて同じ配位環境を持ちます。

📝 Exercise 3: 機械学習で形成エネルギー予測 Hard

**問題** : Materials Projectから遷移金属（Fe、Ni、Cu、Ti、V）の材料データをダウンロードし、密度、バンドギャップ、原子数を特徴量として、形成エネルギーを予測する機械学習モデルを構築してください。Random ForestとGradient Boostingの両方を試し、どちらが精度が高いか評価してください。

💡 解答例
    
    
    from mp_api.client import MPRester
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.metrics import mean_absolute_error, r2_score
    import matplotlib.pyplot as plt
    
    API_KEY = "your_api_key_here"
    
    def download_transition_metal_data(elements, api_key=API_KEY):
        """遷移金属データのダウンロード"""
        all_data = []
    
        with MPRester(api_key) as mpr:
            for element in elements:
                print(f"Downloading {element}...")
                docs = mpr.materials.summary.search(
                    chemsys=element,
                    fields=["material_id", "formula_pretty", "formation_energy_per_atom",
                            "band_gap", "density", "nsites", "is_stable"],
                    num_chunks=1
                )
    
                for doc in docs[:50]:  # 各元素50材料まで
                    if doc.is_stable:
                        all_data.append({
                            'Material ID': doc.material_id,
                            'Formula': doc.formula_pretty,
                            'Density [g/cm³]': doc.density,
                            'Band Gap [eV]': doc.band_gap,
                            'Number of Sites': doc.nsites,
                            'Formation Energy [eV/atom]': doc.formation_energy_per_atom
                        })
    
        df = pd.DataFrame(all_data)
        return df
    
    # データダウンロード
    elements = ['Fe', 'Ni', 'Cu', 'Ti', 'V']
    print("=== Data Download ===")
    # tm_data = download_transition_metal_data(elements)
    
    # 疑似データで代用（実際にはAPIから取得）
    np.random.seed(42)
    n_samples = 150
    tm_data = pd.DataFrame({
        'Material ID': [f'mp-{i}' for i in range(n_samples)],
        'Formula': [f'Material_{i}' for i in range(n_samples)],
        'Density [g/cm³]': np.random.uniform(2, 10, n_samples),
        'Band Gap [eV]': np.random.uniform(0, 3, n_samples),
        'Number of Sites': np.random.randint(2, 20, n_samples),
        'Formation Energy [eV/atom]': np.random.uniform(-2.5, 0.5, n_samples)
    })
    
    print(f"Downloaded {len(tm_data)} materials")
    print(tm_data.head())
    
    # 特徴量とターゲット
    X = tm_data[['Density [g/cm³]', 'Band Gap [eV]', 'Number of Sites']]
    y = tm_data['Formation Energy [eV/atom]']
    
    # Train-Test分割
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"\nTraining: {len(X_train)}, Test: {len(X_test)}")
    
    # モデル訓練
    models = {
        'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, max_depth=5,
                                                         learning_rate=0.1, random_state=42)
    }
    
    results = {}
    
    print("\n=== Model Training ===")
    for name, model in models.items():
        model.fit(X_train, y_train)
    
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
    
        train_mae = mean_absolute_error(y_train, y_train_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        test_r2 = r2_score(y_test, y_test_pred)
    
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
    
        results[name] = {
            'Test MAE': test_mae,
            'Test R²': test_r2,
            'CV R² Mean': cv_scores.mean(),
            'Predictions': y_test_pred
        }
    
        print(f"\n{name}:")
        print(f"  Test MAE: {test_mae:.3f} eV/atom")
        print(f"  Test R²: {test_r2:.3f}")
        print(f"  CV R²: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    
    # 比較
    best_model = max(results, key=lambda k: results[k]['Test R²'])
    print(f"\n=== Best Model: {best_model} ===")
    print(f"Test R² = {results[best_model]['Test R²']:.3f}")
    
    # 可視化
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    for i, (name, model) in enumerate(models.items()):
        y_pred = results[name]['Predictions']
    
        axes[i].scatter(y_test, y_pred, alpha=0.6, s=50)
        axes[i].plot([y_test.min(), y_test.max()],
                      [y_test.min(), y_test.max()],
                      'r--', linewidth=2)
        axes[i].set_xlabel('True Formation Energy [eV/atom]', fontsize=11)
        axes[i].set_ylabel('Predicted [eV/atom]', fontsize=11)
        axes[i].set_title(f'{name}\nR² = {results[name]["Test R²"]:.3f}', fontsize=12)
        axes[i].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('ml_formation_energy_prediction.png', dpi=300, bbox_inches='tight')
    plt.show()
    

**期待される出力** : Random ForestとGradient BoostingのR²スコア比較（通常0.6-0.8程度）

**解説** : 形成エネルギーは密度、バンドギャップ、構造の複雑さ（原子数）と相関します。Random Forestは非線形関係を捉えやすく、Gradient Boostingは細かい誤差を逐次的に修正します。実際のデータでは、より多くの特徴量（電気陰性度、原子半径など）を追加すると精度が向上します。

## 参考文献

**[1]** Jain, A., Ong, S.P., Hautier, G., et al. (2013). _Commentary: The Materials Project: A materials genome approach to accelerating materials innovation_. APL Materials, 1(1), 011002. DOI: 10.1063/1.4812323. 

**[2]** Ong, S.P., Richards, W.D., Jain, A., et al. (2013). _Python Materials Genomics (pymatgen): A robust, open-source python library for materials analysis_. Computational Materials Science, 68, 314-319, pp. 314-319. 

**[3]** Otis, R., Liu, Z.K. (2017). _pycalphad: CALPHAD-based Computational Thermodynamics in Python_. Journal of Open Research Software, 5(1), pp. 1-11. 

**[4]** Ward, L., Agrawal, A., Choudhary, A., Wolverton, C. (2016). _A general-purpose machine learning framework for predicting properties of inorganic materials_. npj Computational Materials, 2, 16028, pp. 1-7. 

**[5]** Pedregosa, F., Varoquaux, G., Gramfort, A., et al. (2011). _Scikit-learn: Machine Learning in Python_. Journal of Machine Learning Research, 12, 2825-2830. 

**[6]** McKinney, W. (2010). _Data Structures for Statistical Computing in Python_. Proceedings of the 9th Python in Science Conference, pp. 56-61. 

**[7]** Hunter, J.D. (2007). _Matplotlib: A 2D Graphics Environment_. Computing in Science & Engineering, 9(3), 90-95. 

## 学習目標の確認

### 本章で習得すべきスキルと知識

#### レベル1: 基本理解（知識）

  * Materials Project APIの基本的な使い方を理解している
  * pymatgenで結晶構造を生成・操作できることを知っている
  * pycalphadが状態図計算に使われることを理解している
  * 機械学習で材料特性を予測する基本的な流れを知っている
  * 材料記述子（特徴量）の概念を理解している
  * 高スループット計算の利点を説明できる

#### レベル2: 実践スキル（応用）

  * Materials Project APIで材料データを検索・ダウンロードできる
  * pymatgenでスーパーセルを作成し、配位数や最近接距離を計算できる
  * pymatgenで対称性解析と空間群決定ができる
  * pandasで材料データを統計解析・可視化できる
  * scikit-learnで回帰モデルを構築し、材料特性を予測できる
  * 材料スクリーニングワークフローを設計・実装できる
  * Pythonで複数材料のバッチ処理を自動化できる

#### レベル3: 高度な応用（問題解決）

  * 大規模材料データベースから目的に応じたデータを効率的に収集できる
  * 複雑な材料記述子を設計し、機械学習モデルの精度を向上できる
  * 異なる機械学習アルゴリズムを比較し、最適なモデルを選択できる
  * 高スループット計算で新規材料候補を発見できる
  * 実験データと計算データを統合した材料設計ができる
  * 独自の材料解析ツールやワークフローを開発できる
  * 研究論文で報告された手法をPythonで再現・拡張できる

**確認方法** : 上記の演習問題（Exercise 1-3）をすべて自力で解けることを確認してください。特に、Exercise 3（機械学習予測）は、実際の研究プロジェクトで頻繁に使用される重要なスキルです。

← Chapter 4: 加工と熱処理（準備中） [シリーズTOPへ →](<index.html>)

### 免責事項

  * 本コンテンツは教育・研究・情報提供のみを目的としており、専門的な助言(法律・会計・技術的保証など)を提供するものではありません。
  * 本コンテンツおよび付随するCode examplesは「現状有姿(AS IS)」で提供され、明示または黙示を問わず、商品性、特定目的適合性、権利非侵害、正確性・完全性、動作・安全性等いかなる保証もしません。
  * 外部リンク、第三者が提供するデータ・ツール・ライブラリ等の内容・可用性・安全性について、作成者および東北大学は一切の責任を負いません。
  * 本コンテンツの利用・実行・解釈により直接的・間接的・付随的・特別・結果的・懲罰的損害が生じた場合でも、適用法で許容される最大限の範囲で、作成者および東北大学は責任を負いません。
  * 本コンテンツの内容は、予告なく変更・更新・提供停止されることがあります。
  * 本コンテンツの著作権・ライセンスは明記された条件(例: CC BY 4.0)に従います。当該ライセンスは通常、無保証条項を含みます。
