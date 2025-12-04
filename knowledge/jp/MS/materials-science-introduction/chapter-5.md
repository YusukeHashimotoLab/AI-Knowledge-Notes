---
title: 第5章：Pythonで学ぶ結晶構造可視化
chapter_title: 第5章：Pythonで学ぶ結晶構造可視化
subtitle: pymatgenライブラリとMaterials Projectの活用
reading_time: 40-45分
difficulty: 中級〜上級
code_examples: 6
---

pymatgenは、材料科学のための強力なPythonライブラリです。結晶構造の作成・操作・解析・可視化を簡単に行えます。この章では、pymatgenの基本的な使い方から、Materials Projectデータベースを活用した実践的な材料解析まで学びます。 

## 学習目標

この章を読むことで、以下を習得できます：

  * ✅ pymatgenのインストールと基本的な使い方を理解する
  * ✅ Structureオブジェクトを作成し、結晶構造を操作できる
  * ✅ CIFファイルから結晶構造を読み込み、情報を抽出できる
  * ✅ Materials Project APIを使って材料データを取得できる
  * ✅ 代表的な材料（Si, Fe, Al₂O₃）の結晶構造を解析できる
  * ✅ 構造可視化から特性予測までの総合ワークフローを実行できる

* * *

## 5.1 pymatgen入門

### pymatgenとは

**pymatgen (Python Materials Genomics)** は、材料科学のための包括的なPythonライブラリです。

**主な機能** ：

  * 結晶構造の作成・操作・解析
  * CIF, POSCAR, XYZなど多様なフォーマットのI/O
  * 結晶対称性の解析
  * 状態図（Phase Diagram）の作成
  * Materials Projectデータベースへのアクセス
  * 電子構造計算（VASP, Quantum Espressoなど）のセットアップ

### インストール

pipを使って簡単にインストールできます：
    
    
    # 基本インストール
    pip install pymatgen
    
    # 可視化機能を含む完全版
    pip install pymatgen[all]
    
    # または、condaを使用する場合
    conda install -c conda-forge pymatgen
    

#### ⚠️ 注意事項

pymatgenは依存パッケージが多いため、インストールに時間がかかることがあります。仮想環境での使用を推奨します。
    
    
    # 仮想環境の作成（推奨）
    python -m venv pymatgen_env
    source pymatgen_env/bin/activate  # Windowsの場合: pymatgen_env\Scripts\activate
    pip install pymatgen
    

### コード例1: pymatgenで単純立方格子を作成

pymatgenの基本操作として、単純立方格子（Simple Cubic）を作成します。
    
    
    from pymatgen.core import Structure, Lattice
    import numpy as np
    
    # 単純立方格子の作成
    # Latticeオブジェクト: 格子ベクトルを定義
    a = 3.0  # 格子定数（Å）
    
    # 方法1: 格子定数から立方晶を作成（最も簡単）
    lattice = Lattice.cubic(a)
    
    print("="*70)
    print("単純立方格子の作成（pymatgen）")
    print("="*70)
    
    print(f"\n【格子情報】")
    print(f"格子定数 a = {a} Å")
    print(f"\n格子ベクトル:")
    print(lattice.matrix)
    print(f"\n格子体積: {lattice.volume:.4f} Å³")
    
    # Structureオブジェクト: 格子 + 原子位置
    # 原子種と分数座標（fractional coordinates）を指定
    species = ["Po"]  # ポロニウム（単純立方の例）
    coords = [[0, 0, 0]]  # 格子の原点に1個の原子
    
    structure = Structure(lattice, species, coords)
    
    print(f"\n【結晶構造情報】")
    print(f"組成式: {structure.composition.reduced_formula}")
    print(f"原子数: {structure.num_sites}")
    print(f"密度: {structure.density:.4f} g/cm³")
    
    # 原子位置の表示
    print(f"\n【原子位置】")
    for i, site in enumerate(structure.sites):
        print(f"原子 {i+1}: {site.specie} @ {site.frac_coords} (分数座標)")
        print(f"        {site.specie} @ {site.coords} Å (デカルト座標)")
    
    # 方法2: より複雑な格子（FCC）を作成
    print("\n" + "="*70)
    print("面心立方格子（FCC）の作成")
    print("="*70)
    
    # FCC格子: 頂点 + 面心に原子
    a_fcc = 3.615  # 銅の格子定数（Å）
    lattice_fcc = Lattice.cubic(a_fcc)
    
    # FCC構造の原子位置（分数座標）
    # 頂点1個 + 面心3個 = 単位格子あたり4原子
    species_fcc = ["Cu"] * 4
    coords_fcc = [
        [0.0, 0.0, 0.0],  # 頂点
        [0.5, 0.5, 0.0],  # xy面の面心
        [0.5, 0.0, 0.5],  # xz面の面心
        [0.0, 0.5, 0.5],  # yz面の面心
    ]
    
    structure_fcc = Structure(lattice_fcc, species_fcc, coords_fcc)
    
    print(f"\n【FCC構造情報（銅）】")
    print(f"組成式: {structure_fcc.composition.reduced_formula}")
    print(f"原子数: {structure_fcc.num_sites}")
    print(f"密度: {structure_fcc.density:.4f} g/cm³（実測値: 8.96 g/cm³）")
    print(f"格子体積: {structure_fcc.volume:.4f} Å³")
    
    # 最近接距離の計算
    print(f"\n【最近接原子間距離】")
    # すべての原子ペアの距離を計算
    distances = []
    for i in range(len(structure_fcc)):
        for j in range(i+1, len(structure_fcc)):
            dist = structure_fcc.get_distance(i, j)
            distances.append(dist)
    
    min_distance = min(distances)
    print(f"最近接距離: {min_distance:.4f} Å")
    print(f"理論値: {a_fcc / np.sqrt(2):.4f} Å（面対角線/2）")
    
    # 配位数の計算（特定の原子の周りの近接原子数）
    neighbors = structure_fcc.get_neighbors(structure_fcc[0], r=min_distance * 1.1)
    print(f"配位数: {len(neighbors)}（FCCの理論値: 12）")
    
    print("\n" + "="*70)
    print("pymatgenの利点:")
    print("- 数行で複雑な結晶構造を作成可能")
    print("- 対称性を自動で処理")
    print("- 格子パラメータと原子位置から密度などを自動計算")
    print("- 距離・角度・配位数などの幾何学的情報を簡単に取得")
    

**解説** : pymatgenでは、`Lattice`オブジェクトで格子を定義し、`Structure`オブジェクトで原子位置を指定することで、結晶構造を作成します。分数座標（格子ベクトルの線形結合）を使うため、対称性を保ちながら操作できます。

* * *

## 5.2 結晶構造の読み込みと表示

### CIFファイルとは

**CIF (Crystallographic Information File)** は、結晶構造データの標準フォーマットです。

**CIFファイルの内容** ：

  * 格子定数（a, b, c, α, β, γ）
  * 空間群（Space Group）
  * 原子種と座標
  * 対称操作
  * その他のメタデータ

CIFファイルは、[Crystallography Open Database (COD)](<https://www.crystallography.net/cod/>)や[Materials Project](<https://materialsproject.org/>)から無料でダウンロードできます。

### コード例2: CIFファイルの読み込みと構造情報の表示

CIFファイルから結晶構造を読み込み、詳細な情報を表示します。
    
    
    from pymatgen.core import Structure
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
    import numpy as np
    
    # CIFファイルの内容を文字列として定義（実際にはファイルから読み込む）
    # 例: シリコン（Si）のCIFデータ
    cif_data_si = """
    data_Si
    _cell_length_a    5.4310
    _cell_length_b    5.4310
    _cell_length_c    5.4310
    _cell_angle_alpha 90.0
    _cell_angle_beta  90.0
    _cell_angle_gamma 90.0
    _space_group_name_H-M_alt 'F d -3 m'
    _space_group_IT_number    227
    loop_
    _atom_site_label
    _atom_site_type_symbol
    _atom_site_fract_x
    _atom_site_fract_y
    _atom_site_fract_z
    Si1 Si 0.000 0.000 0.000
    Si2 Si 0.250 0.250 0.250
    """
    
    # CIF文字列から構造を読み込む
    # 実際のファイルから読み込む場合: Structure.from_file("structure.cif")
    from io import StringIO
    structure_si = Structure.from_str(cif_data_si, fmt="cif")
    
    print("="*70)
    print("CIFファイルから読み込んだ結晶構造（シリコン）")
    print("="*70)
    
    # 基本情報
    print(f"\n【基本情報】")
    print(f"組成式: {structure_si.composition.reduced_formula}")
    print(f"化学式（Hill表記）: {structure_si.composition.hill_formula}")
    print(f"単位格子あたりの原子数: {structure_si.num_sites}")
    print(f"密度: {structure_si.density:.4f} g/cm³（実測値: 2.33 g/cm³）")
    
    # 格子情報
    lattice = structure_si.lattice
    print(f"\n【格子情報】")
    print(f"結晶系: 立方晶系（Cubic）")
    print(f"格子定数:")
    print(f"  a = {lattice.a:.4f} Å")
    print(f"  b = {lattice.b:.4f} Å")
    print(f"  c = {lattice.c:.4f} Å")
    print(f"格子角:")
    print(f"  α = {lattice.alpha:.2f}°")
    print(f"  β = {lattice.beta:.2f}°")
    print(f"  γ = {lattice.gamma:.2f}°")
    print(f"格子体積: {lattice.volume:.4f} Å³")
    
    # 対称性解析
    sga = SpacegroupAnalyzer(structure_si)
    print(f"\n【対称性情報】")
    print(f"空間群記号（Hermann-Mauguin）: {sga.get_space_group_symbol()}")
    print(f"空間群番号: {sga.get_space_group_number()}")
    print(f"点群: {sga.get_point_group_symbol()}")
    print(f"結晶系: {sga.get_crystal_system()}")
    
    # 原子位置の表示
    print(f"\n【原子位置（単位格子内）】")
    for i, site in enumerate(structure_si.sites):
        print(f"\n原子 {i+1}:")
        print(f"  元素: {site.specie}")
        print(f"  分数座標: {site.frac_coords}")
        print(f"  デカルト座標: {site.coords} Å")
    
    # 原始格子（Primitive Cell）への変換
    primitive = structure_si.get_primitive_structure()
    print(f"\n【原始格子への変換】")
    print(f"元の単位格子の原子数: {structure_si.num_sites}")
    print(f"原始格子の原子数: {primitive.num_sites}")
    print(f"原始格子の体積: {primitive.volume:.4f} Å³")
    
    # 従来格子（Conventional Cell）の取得
    conventional = sga.get_conventional_standard_structure()
    print(f"\n従来格子の原子数: {conventional.num_sites}")
    print(f"従来格子の体積: {conventional.volume:.4f} Å³")
    
    # 最近接距離の計算
    print(f"\n【最近接原子間距離】")
    all_distances = []
    for i in range(len(structure_si)):
        for j in range(i+1, len(structure_si)):
            dist = structure_si.get_distance(i, j)
            all_distances.append(dist)
    
    if all_distances:
        min_dist = min(all_distances)
        print(f"最近接距離: {min_dist:.4f} Å")
        print(f"理論値（ダイヤモンド構造）: {lattice.a * np.sqrt(3) / 4:.4f} Å")
    
    # 配位数
    neighbors = structure_si.get_neighbors(structure_si[0], r=min_dist * 1.1)
    print(f"配位数: {len(neighbors)}（ダイヤモンド構造の理論値: 4）")
    
    print("\n" + "="*70)
    print("CIFファイルの利点:")
    print("- 標準フォーマットで結晶構造を保存・共有")
    print("- 格子定数・空間群・原子位置を含む完全な情報")
    print("- pymatgenで簡単に読み込み・解析可能")
    print("- データベース（COD, Materials Projectなど）から入手可能")
    

**解説** : CIFファイルには結晶構造の完全な情報が含まれています。pymatgenを使えば、CIFファイルから構造を読み込み、格子定数、空間群、原子位置などの情報を簡単に抽出できます。`SpacegroupAnalyzer`を使うと対称性の詳細な解析も可能です。

### コード例3: 結晶構造の3D可視化（pymatgen + matplotlib）

pymatgenで作成した結晶構造を3Dで可視化します。
    
    
    from pymatgen.core import Structure, Lattice
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import numpy as np
    
    def visualize_structure_3d(structure, supercell=(1, 1, 1), show_bonds=True, bond_cutoff=3.0):
        """
        pymatgen構造を3Dで可視化
    
        Parameters:
        structure: pymatgen Structure object
        supercell: スーパーセルのサイズ（タプル）
        show_bonds: 結合を表示するか
        bond_cutoff: 結合とみなす距離の閾値（Å）
        """
        # スーパーセルを作成（より大きな領域を可視化）
        structure_super = structure * supercell
    
        # 図の作成
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
    
        # 原子種ごとに色を定義
        element_colors = {
            'Si': '#b8b8d0',  # 薄紫
            'Fe': '#e06633',  # オレンジ
            'Al': '#a8a8a8',  # 灰色
            'O': '#ff0d0d',   # 赤
            'Cu': '#ffa500',  # オレンジ
        }
    
        # 原子種ごとのサイズ
        element_sizes = {
            'Si': 200,
            'Fe': 200,
            'Al': 150,
            'O': 100,
            'Cu': 180,
        }
    
        # 原子をプロット
        for site in structure_super.sites:
            element = str(site.specie)
            color = element_colors.get(element, '#808080')  # デフォルトは灰色
            size = element_sizes.get(element, 150)
    
            x, y, z = site.coords
            ax.scatter(x, y, z, c=color, s=size, edgecolors='black',
                      linewidth=1.5, alpha=0.8, depthshade=True, label=element)
    
        # 結合を描画
        if show_bonds:
            drawn_bonds = set()  # 重複を避けるため
            for i, site1 in enumerate(structure_super.sites):
                neighbors = structure_super.get_neighbors(site1, r=bond_cutoff)
                for neighbor in neighbors:
                    # ペアを識別子として登録（順序不同）
                    pair = tuple(sorted([i, structure_super.sites.index(neighbor.site)]))
                    if pair not in drawn_bonds:
                        drawn_bonds.add(pair)
                        x_vals = [site1.coords[0], neighbor.coords[0]]
                        y_vals = [site1.coords[1], neighbor.coords[1]]
                        z_vals = [site1.coords[2], neighbor.coords[2]]
                        ax.plot(x_vals, y_vals, z_vals, 'k-', linewidth=0.8, alpha=0.3)
    
        # 単位格子の枠を描画
        lattice_vecs = structure.lattice.matrix
        origin = np.array([0, 0, 0])
    
        # 立方体の辺
        vertices = [
            origin,
            lattice_vecs[0],
            lattice_vecs[0] + lattice_vecs[1],
            lattice_vecs[1],
            lattice_vecs[2],
            lattice_vecs[2] + lattice_vecs[0],
            lattice_vecs[2] + lattice_vecs[0] + lattice_vecs[1],
            lattice_vecs[2] + lattice_vecs[1]
        ]
    
        # 底面の辺
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),  # 底面
            (4, 5), (5, 6), (6, 7), (7, 4),  # 上面
            (0, 4), (1, 5), (2, 6), (3, 7)   # 縦の辺
        ]
    
        for edge in edges:
            v1, v2 = vertices[edge[0]], vertices[edge[1]]
            ax.plot([v1[0], v2[0]], [v1[1], v2[1]], [v1[2], v2[2]],
                   'b-', linewidth=2, alpha=0.6)
    
        # 軸ラベル
        ax.set_xlabel('X (Å)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Y (Å)', fontsize=12, fontweight='bold')
        ax.set_zlabel('Z (Å)', fontsize=12, fontweight='bold')
    
        # タイトル
        formula = structure.composition.reduced_formula
        ax.set_title(f'{formula} 結晶構造（{supercell[0]}×{supercell[1]}×{supercell[2]} スーパーセル）',
                    fontsize=14, fontweight='bold', pad=20)
    
        # 凡例（重複を削除）
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), fontsize=11, loc='upper right')
    
        # 視点の調整
        ax.view_init(elev=20, azim=45)
        ax.set_box_aspect([1, 1, 1])
    
        plt.tight_layout()
        plt.show()
    
    # シリコンのダイヤモンド構造を作成
    a_si = 5.431  # Å
    lattice_si = Lattice.cubic(a_si)
    
    # ダイヤモンド構造の原子位置（FCC + 内部原子）
    species_si = ["Si"] * 8
    coords_si = [
        [0.00, 0.00, 0.00],  # FCC頂点
        [0.50, 0.50, 0.00],  # FCC面心
        [0.50, 0.00, 0.50],
        [0.00, 0.50, 0.50],
        [0.25, 0.25, 0.25],  # ダイヤモンド内部
        [0.75, 0.75, 0.25],
        [0.75, 0.25, 0.75],
        [0.25, 0.75, 0.75],
    ]
    
    structure_si = Structure(lattice_si, species_si, coords_si)
    
    print("="*70)
    print("シリコン（Si）のダイヤモンド構造を3D可視化")
    print("="*70)
    print(f"組成式: {structure_si.composition.reduced_formula}")
    print(f"格子定数 a = {a_si} Å")
    print(f"単位格子あたりの原子数: {structure_si.num_sites}")
    print(f"結晶構造: ダイヤモンド型（FCC + 内部原子）")
    print("\n3Dプロットを表示中...")
    
    # 可視化（2x2x2スーパーセルで見やすく）
    visualize_structure_3d(structure_si, supercell=(2, 2, 2), show_bonds=True, bond_cutoff=2.5)
    
    print("\n" + "="*70)
    print("可視化のポイント:")
    print("- 原子を球で表現")
    print("- 結合を線で表現（距離が閾値以下のペア）")
    print("- 単位格子の枠を青線で表示")
    print("- スーパーセルで周期構造を理解しやすく")
    

**解説** : pymatgenのStructureオブジェクトから原子座標を取得し、matplotlibで3D可視化できます。スーパーセル（単位格子を繰り返した構造）を作成することで、結晶の周期性が分かりやすくなります。結合の表示により、配位構造も理解できます。

* * *

## 5.3 Materials Projectデータベース活用

### Materials Projectとは

**Materials Project** は、第一原理計算に基づく材料データベースです。

**主な特徴** ：

  * 140,000種類以上の材料データ
  * 結晶構造、電子構造、熱力学的性質
  * API経由で無料でアクセス可能
  * pymatgenで簡単に利用できる

**含まれるデータ** ：

  * 結晶構造（CIF）
  * バンドギャップ
  * 生成エネルギー
  * 弾性定数
  * 電子状態密度（DOS）
  * バンド構造

### APIキーの取得（オプション）

Materials Project APIを使用するには、[Materials Project](<https://materialsproject.org/>)でアカウントを作成し、APIキーを取得します（無料）。

  1. <https://materialsproject.org/> にアクセス
  2. アカウント登録（Sign Up）
  3. Dashboard → API → Generate API Keyでキーを取得

#### ⚠️ APIキーなしでも使える方法

以下のコード例では、APIキーがない場合でも動作するように、ローカルで構造を作成する方法も示します。実際のプロジェクトでは、APIを使用することで、より多くの材料データにアクセスできます。

### コード例4: Materials Project APIで材料データを取得

Materials Project APIを使って材料データを取得します（APIキーあり・なし両方の例）。
    
    
    from pymatgen.core import Structure
    from pymatgen.ext.matproj import MPRester
    import warnings
    warnings.filterwarnings('ignore')
    
    # Materials Project APIを使った材料データ取得
    def get_material_from_mp(formula, api_key=None):
        """
        Materials Projectから材料データを取得
    
        Parameters:
        formula: 化学式（例: "Si", "Fe", "Al2O3"）
        api_key: Materials Project APIキー（Noneの場合はローカルデータを使用）
    
        Returns:
        structure: pymatgen Structure object
        properties: 材料特性の辞書
        """
        if api_key is not None:
            # APIキーがある場合: Materials Projectから取得
            try:
                with MPRester(api_key) as mpr:
                    # 化学式で検索
                    entries = mpr.get_entries(formula)
    
                    if not entries:
                        print(f"警告: {formula}のデータが見つかりませんでした")
                        return None, None
    
                    # 最も安定な構造（エネルギーが最も低い）を選択
                    entry = min(entries, key=lambda e: e.energy_per_atom)
                    structure = entry.structure
    
                    # 材料IDを取得
                    material_id = entry.entry_id
    
                    # 追加情報を取得
                    material_data = mpr.get_doc(material_id)
    
                    properties = {
                        'formula': structure.composition.reduced_formula,
                        'material_id': material_id,
                        'energy_per_atom': entry.energy_per_atom,
                        'band_gap': material_data.get('band_gap', 'N/A'),
                        'density': structure.density,
                        'space_group': material_data.get('space_group', 'N/A'),
                    }
    
                    return structure, properties
    
            except Exception as e:
                print(f"API取得エラー: {e}")
                print("ローカルデータを使用します")
    
        # APIキーがない場合、またはエラーの場合: ローカルで構造を作成
        print(f"ローカルデータで{formula}の構造を作成します")
    
        if formula == "Si":
            # シリコン（ダイヤモンド構造）
            a = 5.431
            lattice = Lattice.cubic(a)
            species = ["Si"] * 8
            coords = [
                [0.00, 0.00, 0.00], [0.50, 0.50, 0.00],
                [0.50, 0.00, 0.50], [0.00, 0.50, 0.50],
                [0.25, 0.25, 0.25], [0.75, 0.75, 0.25],
                [0.75, 0.25, 0.75], [0.25, 0.75, 0.75],
            ]
            structure = Structure(lattice, species, coords)
            properties = {
                'formula': 'Si',
                'material_id': 'local',
                'energy_per_atom': 'N/A',
                'band_gap': 1.12,  # eV（実測値）
                'density': structure.density,
                'space_group': 'Fd-3m (227)',
            }
    
        elif formula == "Fe":
            # 鉄（BCC構造）
            a = 2.866
            lattice = Lattice.cubic(a)
            species = ["Fe"] * 2
            coords = [[0, 0, 0], [0.5, 0.5, 0.5]]
            structure = Structure(lattice, species, coords)
            properties = {
                'formula': 'Fe',
                'material_id': 'local',
                'energy_per_atom': 'N/A',
                'band_gap': 0.0,  # 金属
                'density': structure.density,
                'space_group': 'Im-3m (229)',
            }
    
        elif formula == "Al2O3":
            # アルミナ（コランダム構造、簡略化）
            a = 4.759
            c = 12.991
            lattice = Lattice.hexagonal(a, c)
            # 簡略化のため、基本的な原子位置のみ
            species = ["Al", "Al", "O", "O", "O"]
            coords = [
                [0.0, 0.0, 0.352],
                [0.0, 0.0, 0.148],
                [0.306, 0.0, 0.25],
                [0.0, 0.306, 0.25],
                [0.694, 0.694, 0.25],
            ]
            structure = Structure(lattice, species, coords)
            properties = {
                'formula': 'Al2O3',
                'material_id': 'local',
                'energy_per_atom': 'N/A',
                'band_gap': 8.8,  # eV（実測値）
                'density': structure.density,
                'space_group': 'R-3c (167)',
            }
    
        else:
            print(f"エラー: {formula}はローカルデータにありません")
            return None, None
    
        return structure, properties
    
    
    # 使用例
    print("="*70)
    print("Materials Projectから材料データを取得")
    print("="*70)
    
    # APIキーを設定（持っている場合）
    # API_KEY = "your_api_key_here"  # 自分のAPIキーに置き換え
    API_KEY = None  # APIキーがない場合
    
    # 複数の材料を取得
    materials = ["Si", "Fe", "Al2O3"]
    
    for formula in materials:
        print(f"\n{'='*70}")
        print(f"【{formula}】")
        print('='*70)
    
        structure, props = get_material_from_mp(formula, api_key=API_KEY)
    
        if structure is None:
            continue
    
        # 構造情報を表示
        print(f"\n組成式: {props['formula']}")
        print(f"Material ID: {props['material_id']}")
        print(f"空間群: {props['space_group']}")
        print(f"バンドギャップ: {props['band_gap']} eV")
        print(f"密度: {props['density']:.4f} g/cm³")
    
        if props['energy_per_atom'] != 'N/A':
            print(f"原子あたりのエネルギー: {props['energy_per_atom']:.4f} eV/atom")
    
        # 格子情報
        lattice = structure.lattice
        print(f"\n格子定数:")
        print(f"  a = {lattice.a:.4f} Å")
        print(f"  b = {lattice.b:.4f} Å")
        print(f"  c = {lattice.c:.4f} Å")
        print(f"格子体積: {lattice.volume:.4f} Å³")
    
        # 原子数
        print(f"\n単位格子あたりの原子数: {structure.num_sites}")
    
        # 原子の内訳
        composition = structure.composition
        print(f"原子組成:")
        for element, count in composition.items():
            print(f"  {element}: {count}")
    
    print("\n" + "="*70)
    print("Materials Projectの活用:")
    print("- 140,000以上の材料データベースにアクセス")
    print("- 第一原理計算に基づく高精度データ")
    print("- バンドギャップ、生成エネルギーなどの物性値")
    print("- pymatgenで簡単にデータ取得・解析")
    print("\nAPIキー取得: https://materialsproject.org/")
    

**解説** : Materials Project APIを使うと、膨大な材料データベースから必要な材料の結晶構造や物性値を取得できます。APIキーがない場合でも、ローカルで主要な材料の構造を作成できます。実際のプロジェクトでは、APIを使用することで効率的にデータを収集できます。

* * *

## 5.4 ケーススタディ：代表的材料の構造解析

### 解析対象材料

3つの重要な材料の結晶構造を詳細に解析します：

  * **シリコン（Si）** : 半導体材料の代表
  * **鉄（Fe）** : 構造材料の基本
  * **アルミナ（Al₂O₃）** : セラミックス材料の代表

### コード例5: 複数材料の結晶構造比較（Si, Fe, Al₂O₃）

3つの材料の結晶構造を作成し、特性を比較します。
    
    
    from pymatgen.core import Structure, Lattice
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
    import matplotlib.pyplot as plt
    import numpy as np
    
    def create_material_structures():
        """
        Si, Fe, Al2O3の構造を作成し、特性を比較
        """
        structures = {}
    
        # 1. シリコン（Si）- ダイヤモンド構造
        a_si = 5.431  # Å
        lattice_si = Lattice.cubic(a_si)
        species_si = ["Si"] * 8
        coords_si = [
            [0.00, 0.00, 0.00], [0.50, 0.50, 0.00],
            [0.50, 0.00, 0.50], [0.00, 0.50, 0.50],
            [0.25, 0.25, 0.25], [0.75, 0.75, 0.25],
            [0.75, 0.25, 0.75], [0.25, 0.75, 0.75],
        ]
        structure_si = Structure(lattice_si, species_si, coords_si)
    
        # 2. 鉄（Fe）- BCC構造
        a_fe = 2.866  # Å
        lattice_fe = Lattice.cubic(a_fe)
        species_fe = ["Fe"] * 2
        coords_fe = [[0, 0, 0], [0.5, 0.5, 0.5]]
        structure_fe = Structure(lattice_fe, species_fe, coords_fe)
    
        # 3. アルミナ（Al2O3）- コランダム構造（六方晶系）
        a_al2o3 = 4.759  # Å
        c_al2o3 = 12.991  # Å
        lattice_al2o3 = Lattice.hexagonal(a_al2o3, c_al2o3)
    
        # コランダム構造の原子位置（簡略版）
        species_al2o3 = ["Al"] * 4 + ["O"] * 6
        coords_al2o3 = [
            # Al原子
            [0.0, 0.0, 0.352],
            [0.667, 0.333, 0.019],
            [0.333, 0.667, 0.685],
            [0.0, 0.0, 0.148],
            # O原子
            [0.306, 0.0, 0.25],
            [0.0, 0.306, 0.25],
            [0.694, 0.694, 0.25],
            [0.973, 0.639, 0.917],
            [0.361, 0.333, 0.917],
            [0.639, 0.0, 0.583],
        ]
        structure_al2o3 = Structure(lattice_al2o3, species_al2o3, coords_al2o3)
    
        structures = {
            'Si': structure_si,
            'Fe': structure_fe,
            'Al2O3': structure_al2o3
        }
    
        return structures
    
    
    def analyze_structure(name, structure):
        """
        結晶構造の詳細な解析
        """
        print(f"\n{'='*70}")
        print(f"【{name}の結晶構造解析】")
        print('='*70)
    
        # 基本情報
        print(f"\n組成式: {structure.composition.reduced_formula}")
        print(f"化学式（Hill表記）: {structure.composition.hill_formula}")
        print(f"単位格子あたりの原子数: {structure.num_sites}")
        print(f"密度: {structure.density:.4f} g/cm³")
    
        # 格子情報
        lattice = structure.lattice
        print(f"\n【格子情報】")
        print(f"格子定数:")
        print(f"  a = {lattice.a:.4f} Å")
        print(f"  b = {lattice.b:.4f} Å")
        print(f"  c = {lattice.c:.4f} Å")
        print(f"格子角:")
        print(f"  α = {lattice.alpha:.2f}°")
        print(f"  β = {lattice.beta:.2f}°")
        print(f"  γ = {lattice.gamma:.2f}°")
        print(f"格子体積: {lattice.volume:.4f} Å³")
    
        # 対称性解析
        try:
            sga = SpacegroupAnalyzer(structure)
            print(f"\n【対称性情報】")
            print(f"空間群記号: {sga.get_space_group_symbol()}")
            print(f"空間群番号: {sga.get_space_group_number()}")
            print(f"点群: {sga.get_point_group_symbol()}")
            print(f"結晶系: {sga.get_crystal_system()}")
        except:
            print("\n対称性解析でエラーが発生しました")
    
        # 最近接距離と配位数
        print(f"\n【幾何学的情報】")
    
        # すべての原子ペアの距離を計算
        all_distances = []
        for i in range(len(structure)):
            for j in range(i+1, len(structure)):
                dist = structure.get_distance(i, j)
                all_distances.append(dist)
    
        if all_distances:
            min_dist = min(all_distances)
            print(f"最近接原子間距離: {min_dist:.4f} Å")
    
            # 配位数（最初の原子について）
            neighbors = structure.get_neighbors(structure[0], r=min_dist * 1.2)
            print(f"配位数（代表例）: {len(neighbors)}")
    
        # 原子組成の内訳
        print(f"\n【原子組成】")
        composition = structure.composition
        for element, count in composition.items():
            atomic_percent = (count / structure.num_sites) * 100
            print(f"  {element}: {count} 原子（{atomic_percent:.1f}%）")
    
        # 充填率の推定（簡易版）
        print(f"\n【充填率の推定】")
        # 原子半径の推定値（Å）
        atomic_radii = {
            'Si': 1.17,
            'Fe': 1.26,
            'Al': 1.43,
            'O': 0.66
        }
    
        total_atomic_volume = 0
        for element, count in composition.items():
            if str(element) in atomic_radii:
                r = atomic_radii[str(element)]
                atomic_vol = (4/3) * np.pi * r**3
                total_atomic_volume += atomic_vol * count
    
        cell_volume = lattice.volume
        packing_fraction = total_atomic_volume / cell_volume
    
        print(f"原子の総体積: {total_atomic_volume:.4f} Ų")
        print(f"単位格子体積: {cell_volume:.4f} Ų")
        print(f"推定充填率: {packing_fraction:.4f} ({packing_fraction*100:.2f}%)")
    
        return {
            'density': structure.density,
            'volume': lattice.volume,
            'num_atoms': structure.num_sites,
            'packing_fraction': packing_fraction,
            'min_distance': min_dist if all_distances else 0,
        }
    
    
    # 構造を作成
    structures = create_material_structures()
    
    # 各材料を解析
    analysis_results = {}
    for name, structure in structures.items():
        results = analyze_structure(name, structure)
        analysis_results[name] = results
    
    # 比較プロット
    print("\n" + "="*70)
    print("材料特性の比較グラフを作成中...")
    print("="*70)
    
    materials = list(analysis_results.keys())
    densities = [analysis_results[m]['density'] for m in materials]
    volumes = [analysis_results[m]['volume'] for m in materials]
    num_atoms = [analysis_results[m]['num_atoms'] for m in materials]
    packing_fractions = [analysis_results[m]['packing_fraction'] for m in materials]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    colors = ['#b8b8d0', '#e06633', '#808080']
    
    # 密度の比較
    ax1 = axes[0, 0]
    bars1 = ax1.bar(materials, densities, color=colors, edgecolor='black', linewidth=1.5, alpha=0.7)
    ax1.set_ylabel('密度 (g/cm³)', fontsize=12, fontweight='bold')
    ax1.set_title('密度の比較', fontsize=13, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars1, densities):
        ax1.text(bar.get_x() + bar.get_width()/2, val + 0.2,
                f'{val:.2f}', ha='center', fontsize=11, fontweight='bold')
    
    # 単位格子体積の比較
    ax2 = axes[0, 1]
    bars2 = ax2.bar(materials, volumes, color=colors, edgecolor='black', linewidth=1.5, alpha=0.7)
    ax2.set_ylabel('単位格子体積 (ų)', fontsize=12, fontweight='bold')
    ax2.set_title('単位格子体積の比較', fontsize=13, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars2, volumes):
        ax2.text(bar.get_x() + bar.get_width()/2, val + 5,
                f'{val:.1f}', ha='center', fontsize=11, fontweight='bold')
    
    # 単位格子あたりの原子数
    ax3 = axes[1, 0]
    bars3 = ax3.bar(materials, num_atoms, color=colors, edgecolor='black', linewidth=1.5, alpha=0.7)
    ax3.set_ylabel('原子数', fontsize=12, fontweight='bold')
    ax3.set_title('単位格子あたりの原子数', fontsize=13, fontweight='bold')
    ax3.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars3, num_atoms):
        ax3.text(bar.get_x() + bar.get_width()/2, val + 0.2,
                f'{val}', ha='center', fontsize=11, fontweight='bold')
    
    # 充填率
    ax4 = axes[1, 1]
    bars4 = ax4.bar(materials, [pf*100 for pf in packing_fractions],
                   color=colors, edgecolor='black', linewidth=1.5, alpha=0.7)
    ax4.set_ylabel('充填率 (%)', fontsize=12, fontweight='bold')
    ax4.set_title('推定充填率の比較', fontsize=13, fontweight='bold')
    ax4.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars4, packing_fractions):
        ax4.text(bar.get_x() + bar.get_width()/2, val*100 + 1,
                f'{val*100:.1f}%', ha='center', fontsize=11, fontweight='bold')
    
    plt.suptitle('代表的材料の結晶構造比較（Si, Fe, Al₂O₃）',
                fontsize=15, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.show()
    
    print("\n" + "="*70)
    print("比較のまとめ:")
    print("="*70)
    print("\n【シリコン（Si）】")
    print("- ダイヤモンド構造（立方晶系）")
    print("- 半導体材料（バンドギャップ 1.12 eV）")
    print("- 配位数4（共有結合性）")
    print("- 集積回路、太陽電池に使用")
    
    print("\n【鉄（Fe）】")
    print("- BCC構造（立方晶系）")
    print("- 金属（導体）")
    print("- 配位数8")
    print("- 構造材料の基本（鋼の主成分）")
    
    print("\n【アルミナ（Al₂O₃）】")
    print("- コランダム構造（六方晶系）")
    print("- 絶縁体（バンドギャップ 8.8 eV）")
    print("- 高硬度セラミックス")
    print("- 耐摩耗材料、研磨材、絶縁材に使用")
    

**解説** : pymatgenを使うと、異なる結晶系（立方晶、六方晶）の材料を統一的に扱えます。密度、格子体積、充填率などを計算・比較することで、材料特性の違いが定量的に理解できます。

### コード例6: 総合ワークフロー（構造読み込み→解析→可視化→特性予測）

pymatgenを使った材料解析の総合ワークフローを実行します。
    
    
    from pymatgen.core import Structure, Lattice
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
    from pymatgen.analysis.structure_matcher import StructureMatcher
    import matplotlib.pyplot as plt
    import numpy as np
    
    class MaterialAnalysisPipeline:
        """
        材料解析の総合ワークフローを実行するクラス
        """
    
        def __init__(self, structure, material_name="Unknown"):
            """
            Parameters:
            structure: pymatgen Structure object
            material_name: 材料名
            """
            self.structure = structure
            self.material_name = material_name
            self.analysis_results = {}
    
        def step1_basic_info(self):
            """
            ステップ1: 基本情報の抽出
            """
            print(f"\n{'='*70}")
            print(f"ステップ1: 基本情報の抽出【{self.material_name}】")
            print('='*70)
    
            s = self.structure
            lattice = s.lattice
    
            self.analysis_results['basic'] = {
                'formula': s.composition.reduced_formula,
                'num_sites': s.num_sites,
                'density': s.density,
                'volume': lattice.volume,
                'lattice_a': lattice.a,
                'lattice_b': lattice.b,
                'lattice_c': lattice.c,
            }
    
            print(f"組成式: {self.analysis_results['basic']['formula']}")
            print(f"原子数: {self.analysis_results['basic']['num_sites']}")
            print(f"密度: {self.analysis_results['basic']['density']:.4f} g/cm³")
            print(f"格子体積: {self.analysis_results['basic']['volume']:.4f} ų")
    
            return self.analysis_results['basic']
    
        def step2_symmetry_analysis(self):
            """
            ステップ2: 対称性の解析
            """
            print(f"\n{'='*70}")
            print(f"ステップ2: 対称性の解析")
            print('='*70)
    
            try:
                sga = SpacegroupAnalyzer(self.structure)
    
                self.analysis_results['symmetry'] = {
                    'space_group_symbol': sga.get_space_group_symbol(),
                    'space_group_number': sga.get_space_group_number(),
                    'point_group': sga.get_point_group_symbol(),
                    'crystal_system': sga.get_crystal_system(),
                }
    
                print(f"空間群: {self.analysis_results['symmetry']['space_group_symbol']} "
                      f"(No. {self.analysis_results['symmetry']['space_group_number']})")
                print(f"点群: {self.analysis_results['symmetry']['point_group']}")
                print(f"結晶系: {self.analysis_results['symmetry']['crystal_system']}")
    
            except Exception as e:
                print(f"対称性解析でエラー: {e}")
                self.analysis_results['symmetry'] = None
    
            return self.analysis_results['symmetry']
    
        def step3_geometric_analysis(self):
            """
            ステップ3: 幾何学的解析（距離、配位数）
            """
            print(f"\n{'='*70}")
            print(f"ステップ3: 幾何学的解析")
            print('='*70)
    
            # 最近接距離の計算
            all_distances = []
            for i in range(len(self.structure)):
                for j in range(i+1, len(self.structure)):
                    dist = self.structure.get_distance(i, j)
                    all_distances.append(dist)
    
            if all_distances:
                min_dist = min(all_distances)
                avg_dist = np.mean(all_distances)
    
                # 配位数（最初の原子について）
                neighbors = self.structure.get_neighbors(self.structure[0], r=min_dist * 1.2)
                coordination_number = len(neighbors)
    
                self.analysis_results['geometry'] = {
                    'min_distance': min_dist,
                    'avg_distance': avg_dist,
                    'coordination_number': coordination_number,
                }
    
                print(f"最近接原子間距離: {min_dist:.4f} Å")
                print(f"平均原子間距離: {avg_dist:.4f} Å")
                print(f"配位数（代表例）: {coordination_number}")
    
            else:
                self.analysis_results['geometry'] = None
    
            return self.analysis_results['geometry']
    
        def step4_predict_properties(self):
            """
            ステップ4: 材料特性の予測（経験則ベース）
            """
            print(f"\n{'='*70}")
            print(f"ステップ4: 材料特性の予測")
            print('='*70)
    
            predictions = {}
    
            # 密度から機械的性質を推定（超簡易版）
            density = self.analysis_results['basic']['density']
    
            if density < 3.0:
                material_class = "軽量材料"
                strength_estimate = "低〜中程度"
            elif density < 6.0:
                material_class = "中重量材料"
                strength_estimate = "中〜高"
            else:
                material_class = "重量材料"
                strength_estimate = "高"
    
            predictions['material_class'] = material_class
            predictions['strength_estimate'] = strength_estimate
    
            # 配位数から結合の性質を推定
            if self.analysis_results['geometry']:
                cn = self.analysis_results['geometry']['coordination_number']
    
                if cn == 4:
                    bonding_type = "共有結合性（ダイヤモンド型）"
                    expected_properties = "半導体、硬い、脆い"
                elif cn == 6:
                    bonding_type = "イオン結合性または八面体配位"
                    expected_properties = "セラミックス、硬い、絶縁体"
                elif cn in [8, 12]:
                    bonding_type = "金属結合（高配位数）"
                    expected_properties = "金属、延性、導電性"
                else:
                    bonding_type = "不明"
                    expected_properties = "詳細解析が必要"
    
                predictions['bonding_type'] = bonding_type
                predictions['expected_properties'] = expected_properties
    
            print(f"材料分類: {predictions['material_class']}")
            print(f"推定強度: {predictions['strength_estimate']}")
    
            if 'bonding_type' in predictions:
                print(f"結合の性質: {predictions['bonding_type']}")
                print(f"期待される特性: {predictions['expected_properties']}")
    
            self.analysis_results['predictions'] = predictions
            return predictions
    
        def step5_visualize_summary(self):
            """
            ステップ5: 解析結果の可視化
            """
            print(f"\n{'='*70}")
            print(f"ステップ5: 解析結果の可視化")
            print('='*70)
    
            # サマリープロット
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
            # 格子パラメータ
            ax1 = axes[0, 0]
            lattice_params = [
                self.analysis_results['basic']['lattice_a'],
                self.analysis_results['basic']['lattice_b'],
                self.analysis_results['basic']['lattice_c']
            ]
            ax1.bar(['a', 'b', 'c'], lattice_params, color=['#1f77b4', '#ff7f0e', '#2ca02c'],
                   edgecolor='black', linewidth=1.5, alpha=0.7)
            ax1.set_ylabel('格子定数 (Å)', fontsize=11, fontweight='bold')
            ax1.set_title('格子定数', fontsize=12, fontweight='bold')
            ax1.grid(axis='y', alpha=0.3)
    
            # 基本特性
            ax2 = axes[0, 1]
            properties = ['原子数', '密度\n(g/cm³)', '体積\n(ų)']
            values = [
                self.analysis_results['basic']['num_sites'],
                self.analysis_results['basic']['density'],
                self.analysis_results['basic']['volume']
            ]
            ax2.bar(properties, values, color=['#d62728', '#9467bd', '#8c564b'],
                   edgecolor='black', linewidth=1.5, alpha=0.7)
            ax2.set_ylabel('値', fontsize=11, fontweight='bold')
            ax2.set_title('基本特性', fontsize=12, fontweight='bold')
            ax2.grid(axis='y', alpha=0.3)
    
            # 幾何学的情報
            ax3 = axes[1, 0]
            if self.analysis_results['geometry']:
                geom_labels = ['最近接距離\n(Å)', '平均距離\n(Å)', '配位数']
                geom_values = [
                    self.analysis_results['geometry']['min_distance'],
                    self.analysis_results['geometry']['avg_distance'],
                    self.analysis_results['geometry']['coordination_number']
                ]
                ax3.bar(geom_labels, geom_values, color=['#e377c2', '#7f7f7f', '#bcbd22'],
                       edgecolor='black', linewidth=1.5, alpha=0.7)
                ax3.set_ylabel('値', fontsize=11, fontweight='bold')
                ax3.set_title('幾何学的情報', fontsize=12, fontweight='bold')
                ax3.grid(axis='y', alpha=0.3)
    
            # サマリーテキスト
            ax4 = axes[1, 1]
            ax4.axis('off')
    
            summary_text = f"【{self.material_name} 解析サマリー】\n\n"
            summary_text += f"組成式: {self.analysis_results['basic']['formula']}\n"
    
            if self.analysis_results['symmetry']:
                summary_text += f"空間群: {self.analysis_results['symmetry']['space_group_symbol']}\n"
                summary_text += f"結晶系: {self.analysis_results['symmetry']['crystal_system']}\n"
    
            summary_text += f"\n密度: {self.analysis_results['basic']['density']:.4f} g/cm³\n"
            summary_text += f"単位格子体積: {self.analysis_results['basic']['volume']:.2f} ų\n"
    
            if self.analysis_results['geometry']:
                summary_text += f"\n配位数: {self.analysis_results['geometry']['coordination_number']}\n"
    
            if self.analysis_results['predictions']:
                summary_text += f"\n【予測される特性】\n"
                summary_text += f"{self.analysis_results['predictions']['material_class']}\n"
                if 'expected_properties' in self.analysis_results['predictions']:
                    summary_text += f"{self.analysis_results['predictions']['expected_properties']}\n"
    
            ax4.text(0.1, 0.9, summary_text, fontsize=11, verticalalignment='top',
                    fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
            plt.suptitle(f'{self.material_name} 構造解析結果',
                        fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.show()
    
        def run_full_pipeline(self):
            """
            全ステップを実行
            """
            print(f"\n{'#'*70}")
            print(f"#  材料解析パイプライン開始: {self.material_name}")
            print(f"{'#'*70}")
    
            self.step1_basic_info()
            self.step2_symmetry_analysis()
            self.step3_geometric_analysis()
            self.step4_predict_properties()
            self.step5_visualize_summary()
    
            print(f"\n{'#'*70}")
            print(f"#  解析完了: {self.material_name}")
            print(f"{'#'*70}\n")
    
            return self.analysis_results
    
    
    # 使用例: シリコンの解析
    a_si = 5.431
    lattice_si = Lattice.cubic(a_si)
    species_si = ["Si"] * 8
    coords_si = [
        [0.00, 0.00, 0.00], [0.50, 0.50, 0.00],
        [0.50, 0.00, 0.50], [0.00, 0.50, 0.50],
        [0.25, 0.25, 0.25], [0.75, 0.75, 0.25],
        [0.75, 0.25, 0.75], [0.25, 0.75, 0.75],
    ]
    structure_si = Structure(lattice_si, species_si, coords_si)
    
    # パイプラインの実行
    pipeline = MaterialAnalysisPipeline(structure_si, material_name="シリコン（Si）")
    results = pipeline.run_full_pipeline()
    
    print("\n" + "="*70)
    print("総合ワークフローの利点:")
    print("="*70)
    print("- 系統的な解析（基本→対称性→幾何学→予測→可視化）")
    print("- 再現可能な手順")
    print("- 複数の材料に同じ手法を適用可能")
    print("- 結果を自動的に可視化")
    print("- 経験則に基づく特性予測")
    

**解説** : 総合ワークフローを構築することで、材料解析を系統的かつ効率的に実行できます。基本情報の抽出から対称性解析、幾何学的解析、特性予測、可視化まで、一連の流れを自動化できます。このパイプラインは、新しい材料に対しても同様に適用できます。

* * *

## 5.5 トラブルシューティング

### よくあるエラーと解決法

#### ⚠️ ImportError: No module named 'pymatgen'

**原因** : pymatgenがインストールされていない

**解決法** :
    
    
    pip install pymatgen

#### ⚠️ ValueError: The reduced_formula could not be parsed

**原因** : 化学式の表記が不正

**解決法** : 化学式は正しい表記（例: "Al2O3"、"SiO2"）を使用してください

#### ⚠️ MPRestError: API key is not set

**原因** : Materials Project APIキーが設定されていない

**解決法** :

  1. [Materials Project](<https://materialsproject.org/>)でアカウント作成
  2. APIキーを取得
  3. コードで設定: `MPRester("your_api_key")`

#### ⚠️ 可視化が表示されない

**原因** : matplotlibのバックエンド設定

**解決法** :
    
    
    import matplotlib
    matplotlib.use('TkAgg')  # またはQt5Agg
    import matplotlib.pyplot as plt

* * *

## 5.6 本章のまとめ

### 学んだこと

  1. **pymatgenの基礎**
     * LatticeとStructureオブジェクトの作成
     * 分数座標とデカルト座標の扱い
     * 密度、距離、配位数などの自動計算
  2. **CIFファイルの活用**
     * 標準フォーマットでの構造データの読み書き
     * 格子定数、空間群、原子位置の抽出
     * 対称性解析（SpacegroupAnalyzer）
  3. **3D可視化**
     * matplotlibによる結晶構造の3Dプロット
     * スーパーセルでの周期構造の表示
     * 原子と結合の可視化
  4. **Materials Projectの活用**
     * 140,000以上の材料データベースへのアクセス
     * API経由での材料データ取得
     * バンドギャップ、生成エネルギーなどの物性値
  5. **実践的な材料解析**
     * Si, Fe, Al₂O₃の構造比較
     * 総合ワークフロー（構造→解析→可視化→予測）
     * 経験則に基づく特性予測

### 重要なポイント

  * pymatgenは材料科学のための**包括的なPythonライブラリ**
  * 結晶構造の作成・操作・解析が数行で可能
  * CIFファイルは結晶構造の標準フォーマット
  * Materials Projectは第一原理計算に基づく高精度データベース
  * 系統的なワークフローで効率的な材料解析が可能

### 次のステップ

この材料科学入門シリーズを完了しました。今後の学習として：

  * **電子構造計算** : VASP, Quantum Espressoなどの第一原理計算ソフトウェア
  * **機械学習との融合** : Materials Informaticsによる材料探索
  * **状態図の作成** : pymatgenのPhase Diagram機能
  * **欠陥解析** : 点欠陥、転位、粒界のモデリング
  * **表面科学** : 表面構造とスラブモデル

### 参考資料

  * [pymatgen公式ドキュメント](<https://pymatgen.org/>)
  * [Materials Project](<https://materialsproject.org/>)
  * [Crystallography Open Database (COD)](<https://www.crystallography.net/cod/>)
  * [Materials Project Wiki](<https://wiki.materialsproject.org/Getting_Started>)
