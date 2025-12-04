---
title: 第5章：pymatgenによる結晶学計算実践
chapter_title: 第5章：pymatgenによる結晶学計算実践
subtitle: Materials Projectとの連携から実践的ワークフローまで
reading_time: 30-36分
code_examples: 8
---

🌐 JP | [🇬🇧 EN](<../../../en/MS/crystallography-introduction/chapter-5.html>) | Last sync: 2025-11-16

[AI寺子屋トップ](<../../index.html>) > [MS Dojo](<../index.html>) > [結晶学入門](<index.html>) > 第5章 

## 学習目標

この最終章を学ぶことで、以下の実践的スキルを習得できます：

  * **pymatgenライブラリ** の基本的な使い方を理解し、Structureオブジェクトを操作できる
  * **CIFファイル** の読み込みと書き出しができる
  * **Materials Project API** を使ってデータベースから結晶構造を取得できる
  * **結晶構造の変換** （超格子、スラブモデル）を実行できる
  * **対称性解析** （空間群同定、Wyckoff位置）を実施できる
  * **結晶構造の可視化** （2D/3D）ができる
  * **実践的ワークフロー** （データ取得→解析→可視化）を実行できる
  * Materials Informatics研究への**確実な基盤** を築く

## 1\. pymatgenの概要と環境構築

### 1.1 pymatgenとは

**pymatgen（Python Materials Genomics）** は、材料科学のための包括的なPythonライブラリです。 MIT Materials Project研究グループによって開発され、結晶構造解析、第一原理計算、Materials Informatics研究で広く使われています。 

#### pymatgenの主要機能

  * **結晶構造の表現と操作** ：Structureオブジェクトによる原子配置の管理
  * **CIFファイルの読み書き** ：結晶構造データベースとの連携
  * **対称性解析** ：空間群の自動認識、Wyckoff位置の計算
  * **Materials Project連携** ：14万以上の材料データベースへのアクセス
  * **構造変換** ：超格子、表面モデル、ドーピング
  * **物性計算** ：XRDパターン、状態密度、バンド構造
  * **第一原理計算との連携** ：VASP、Quantum ESPRESSO等の入出力

### 1.2 インストールとセットアップ

#### インストール手順
    
    
    # ターミナル/コマンドプロンプトで実行
    # 基本的なインストール
    pip install pymatgen
    
    # Materials Project API連携用（推奨）
    pip install mp-api
    
    # 可視化ライブラリも一緒にインストール
    pip install matplotlib numpy scipy plotly
    
    # インストール確認
    python -c "import pymatgen; print(pymatgen.__version__)"
    # 出力例: 2024.10.3
    

#### Materials Project APIキーの取得

Materials Projectのデータベースにアクセスするには、無料のAPIキーが必要です：

  1. [Materials Project公式サイト](<https://next-gen.materialsproject.org/>)にアクセス
  2. 右上の「Sign Up」から無料アカウントを作成
  3. ログイン後、「API」セクションに移動
  4. 「Generate API Key」をクリックして新しいAPIキーを生成
  5. 表示されたAPIキー（例: `abc123xyz...`）をコピー
  6. 環境変数として設定するか、スクリプト内で使用

**セキュリティ上の注意** ：APIキーは秘密情報です。GitHubなどにコミットしないでください。

## 2\. Structureオブジェクトの基本操作

### 2.1 結晶構造の作成と情報取得

#### コード例1：Structureオブジェクトの基本操作

シリコン（Si）のダイヤモンド構造を作成し、基本情報を取得します：
    
    
    from pymatgen.core import Structure, Lattice
    import numpy as np
    
    # ダイヤモンド構造のシリコンを作成
    # 格子ベクトルの定義（立方晶、a = 5.43 Å）
    a = 5.43
    lattice = Lattice.cubic(a)
    
    # 原子の種類と分数座標
    species = ['Si'] * 8
    coords = [
        [0.00, 0.00, 0.00],  # FCC格子点
        [0.50, 0.50, 0.00],
        [0.50, 0.00, 0.50],
        [0.00, 0.50, 0.50],
        [0.25, 0.25, 0.25],  # 四面体空隙
        [0.75, 0.75, 0.25],
        [0.75, 0.25, 0.75],
        [0.25, 0.75, 0.75]
    ]
    
    # Structureオブジェクトの作成
    si_structure = Structure(lattice, species, coords)
    
    print("=== シリコン（Si）の結晶構造情報 ===\n")
    
    # 基本情報の取得
    print(f"化学組成: {si_structure.composition}")
    print(f"化学式: {si_structure.formula}")
    print(f"単位格子の体積: {si_structure.volume:.4f} Å³")
    print(f"密度: {si_structure.density:.4f} g/cm³")
    print(f"原子数: {len(si_structure)}")
    
    # 格子定数の取得
    print(f"\n格子定数:")
    print(f"  a = {si_structure.lattice.a:.4f} Å")
    print(f"  b = {si_structure.lattice.b:.4f} Å")
    print(f"  c = {si_structure.lattice.c:.4f} Å")
    print(f"  α = {si_structure.lattice.alpha:.2f}°")
    print(f"  β = {si_structure.lattice.beta:.2f}°")
    print(f"  γ = {si_structure.lattice.gamma:.2f}°")
    
    # 各原子の情報
    print(f"\n原子座標:")
    print(f"{'原子':<8} {'分数座標 (x, y, z)':<30} {'デカルト座標 (Å)':<25}")
    print("-" * 65)
    
    for i, site in enumerate(si_structure):
        species = site.species_string
        frac_coords = site.frac_coords
        cart_coords = site.coords
    
        print(f"{species:<8} ({frac_coords[0]:6.3f}, {frac_coords[1]:6.3f}, {frac_coords[2]:6.3f})    "
              f"({cart_coords[0]:6.3f}, {cart_coords[1]:6.3f}, {cart_coords[2]:6.3f})")
    
    # 最近接原子間距離
    print(f"\n最近接原子間距離:")
    neighbors = si_structure.get_neighbors(si_structure[0], 3.0)
    for neighbor, distance in neighbors:
        print(f"  {distance:.4f} Å")
    

### 2.2 CIFファイルの読み込みと書き出し

**CIF（Crystallographic Information File）** は、結晶構造データの標準フォーマットです。 世界中のデータベース（ICSD、COD、Materials Projectなど）がCIF形式でデータを提供しています。 

#### コード例2：CIFファイルの読み込みと書き出し
    
    
    from pymatgen.core import Structure
    from pymatgen.io.cif import CifWriter
    import os
    
    # 前のコード例で作成したsi_structureを使用
    
    # 1. CIFファイルとして保存
    cif_filename = "Si_diamond.cif"
    cif_writer = CifWriter(si_structure)
    cif_writer.write_file(cif_filename)
    
    print(f"=== CIFファイルの書き出し ===")
    print(f"ファイル名: {cif_filename}")
    print(f"ファイルサイズ: {os.path.getsize(cif_filename)} bytes\n")
    
    # CIFファイルの内容を表示
    print("=== CIFファイルの内容（抜粋） ===")
    with open(cif_filename, 'r') as f:
        lines = f.readlines()
        for line in lines[:30]:  # 最初の30行を表示
            print(line.rstrip())
    
    # 2. CIFファイルの読み込み
    loaded_structure = Structure.from_file(cif_filename)
    
    print("\n=== CIFファイルから読み込んだ構造 ===")
    print(f"化学式: {loaded_structure.formula}")
    print(f"空間群: {loaded_structure.get_space_group_info()}")
    print(f"格子定数 a: {loaded_structure.lattice.a:.4f} Å")
    print(f"原子数: {len(loaded_structure)}")
    
    # 3. オンラインデータベースからCIFファイルを取得する例
    # 例: Crystallography Open Database (COD)
    # 以下はCODからのダウンロード例（実際にはwgetやcurlを使用）
    
    print("\n=== オンラインデータベースからの取得例 ===")
    print("Crystallography Open Database (COD):")
    print("  URL: http://www.crystallography.net/cod/")
    print("  例: Si (COD ID: 9008565)")
    print("  ダウンロード: wget http://www.crystallography.net/cod/9008565.cif")
    
    print("\nInorganic Crystal Structure Database (ICSD):")
    print("  URL: https://icsd.fiz-karlsruhe.de/")
    print("  注: ライセンスが必要（大学・研究機関経由でアクセス可能な場合が多い）")
    
    # 4. 構造の比較
    print("\n=== 元の構造と読み込んだ構造の比較 ===")
    print(f"同一性チェック: {si_structure == loaded_structure}")
    print(f"格子定数の差 (Å): {abs(si_structure.lattice.a - loaded_structure.lattice.a):.6f}")
    
    # クリーンアップ
    # os.remove(cif_filename)  # 必要に応じてコメントアウトを外す
    

## 3\. Materials Project APIの使用

### 3.1 Materials Projectとは

**Materials Project** は、機械学習と第一原理計算を組み合わせた世界最大級の材料データベースです。 14万以上の無機材料の構造、物性、電子状態などのデータが無料で公開されています。 
    
    
    ```mermaid
    flowchart LR
                    A[Materials Projectデータベース] --> B[API経由でアクセス]
                    B --> C[pymatgenで構造取得]
                    C --> D[ローカルで解析]
                    D --> E[可視化・計算]
                    E --> F[研究成果]
    
                    style A fill:#e3f2fd
                    style B fill:#fff3e0
                    style C fill:#e8f5e9
                    style D fill:#fce4ec
                    style E fill:#f3e5f5
                    style F fill:#e0f2f1
    ```

### 3.2 Materials Project APIを使った構造取得

#### コード例3：Materials Project APIによる構造取得

APIキーを使ってシリコンの結晶構造を取得します：
    
    
    from mp_api.client import MPRester
    from pymatgen.core import Structure
    
    # APIキーの設定（ここでは環境変数から取得する方法を推奨）
    import os
    
    # 方法1: 環境変数から取得（推奨）
    API_KEY = os.environ.get('MP_API_KEY', None)
    
    # 方法2: 直接指定（テスト用のみ、本番環境では使わない）
    if API_KEY is None:
        print("警告: MP_API_KEYが環境変数に設定されていません。")
        print("以下のコードはダミーの例です。実際に実行するにはAPIキーが必要です。\n")
        # API_KEY = "your_api_key_here"  # 実際のAPIキーに置き換え
    
    def get_structure_from_mp(api_key=None, demo_mode=True):
        """
        Materials Projectから構造を取得
    
        Parameters:
        -----------
        api_key : str
            Materials Project APIキー
        demo_mode : bool
            デモモード（APIキーなしでダミーデータを返す）
        """
    
        if demo_mode or api_key is None:
            print("=== デモモード: ローカルでSi構造を作成 ===\n")
            # ダミーのSi構造を返す
            lattice = Lattice.cubic(5.43)
            species = ['Si'] * 8
            coords = [
                [0.00, 0.00, 0.00], [0.50, 0.50, 0.00],
                [0.50, 0.00, 0.50], [0.00, 0.50, 0.50],
                [0.25, 0.25, 0.25], [0.75, 0.75, 0.25],
                [0.75, 0.25, 0.75], [0.25, 0.75, 0.75]
            ]
            structure = Structure(lattice, species, coords)
            print(f"Materials Project ID (例): mp-149")
            print(f"化学式: {structure.formula}")
            print(f"空間群: Fd-3m (227)")
            return structure
    
        # 実際のAPI使用
        with MPRester(api_key) as mpr:
            print("=== Materials Projectから構造を取得 ===\n")
    
            # 方法1: Material IDを指定して取得
            mp_id = "mp-149"  # Si (シリコン)
            structure = mpr.get_structure_by_material_id(mp_id)
    
            print(f"Material ID: {mp_id}")
            print(f"化学式: {structure.formula}")
            print(f"空間群: {structure.get_space_group_info()}")
            print(f"格子定数 a: {structure.lattice.a:.4f} Å")
            print(f"密度: {structure.density:.4f} g/cm³")
    
            # 方法2: 化学式で検索
            print("\n=== 化学式による検索例 ===")
            docs = mpr.materials.summary.search(
                formula="Fe2O3",
                fields=["material_id", "formula_pretty", "symmetry"]
            )
    
            print(f"Fe2O3の検索結果: {len(docs)}件")
            for doc in docs[:3]:  # 最初の3件を表示
                print(f"  - {doc.material_id}: {doc.formula_pretty}, "
                      f"空間群 {doc.symmetry.symbol}")
    
            return structure
    
    # 実行（デモモード）
    si_structure_mp = get_structure_from_mp(api_key=API_KEY, demo_mode=(API_KEY is None))
    
    print("\n=== 取得した構造の詳細 ===")
    print(si_structure_mp)
    
    # APIキーの取得方法を案内
    if API_KEY is None:
        print("\n" + "="*60)
        print("Materials Project APIキーの取得方法:")
        print("1. https://next-gen.materialsproject.org/ にアクセス")
        print("2. 右上の 'Sign Up' から無料アカウントを作成")
        print("3. ログイン後、'API' セクションでAPIキーを生成")
        print("4. 環境変数に設定:")
        print("   export MP_API_KEY='your_api_key_here'  # Linux/Mac")
        print("   set MP_API_KEY=your_api_key_here       # Windows")
        print("="*60)
    

### 3.3 Materials Projectでの材料検索

#### コード例4：条件を指定した材料検索

バンドギャップや結晶系などの条件で材料を検索します：
    
    
    from mp_api.client import MPRester
    import pandas as pd
    
    def search_materials(api_key=None, demo_mode=True):
        """
        Materials Projectで条件を指定して材料を検索
        """
    
        if demo_mode or api_key is None:
            print("=== デモモード: サンプル検索結果 ===\n")
    
            # ダミーの検索結果
            data = {
                'material_id': ['mp-149', 'mp-13', 'mp-66'],
                'formula': ['Si', 'GaAs', 'Ge'],
                'band_gap (eV)': [1.15, 1.42, 0.74],
                'formation_energy (eV/atom)': [-0.31, -0.48, -0.28],
                'space_group': ['Fd-3m', 'F-43m', 'Fd-3m']
            }
    
            df = pd.DataFrame(data)
            print("検索条件: バンドギャップ 0.5 - 2.0 eV の半導体")
            print(df.to_string(index=False))
            return df
    
        # 実際のAPI使用
        with MPRester(api_key) as mpr:
            print("=== Materials Projectで材料を検索 ===\n")
    
            # 例1: バンドギャップで検索
            print("検索条件1: バンドギャップ 1.0 - 2.0 eV の半導体")
    
            docs = mpr.materials.summary.search(
                band_gap=(1.0, 2.0),
                num_elements=(1, 2),  # 1-2元素系
                fields=["material_id", "formula_pretty", "band_gap",
                       "formation_energy_per_atom", "symmetry"]
            )
    
            results = []
            for doc in docs[:10]:  # 最初の10件
                results.append({
                    'material_id': doc.material_id,
                    'formula': doc.formula_pretty,
                    'band_gap (eV)': f"{doc.band_gap:.3f}",
                    'formation_energy (eV/atom)': f"{doc.formation_energy_per_atom:.3f}",
                    'space_group': doc.symmetry.symbol
                })
    
            df = pd.DataFrame(results)
            print(df.to_string(index=False))
    
            # 例2: 特定の元素を含む材料の検索
            print("\n検索条件2: Liを含むイオン伝導体候補")
    
            docs = mpr.materials.summary.search(
                elements=["Li", "O"],  # LiとOを含む
                num_elements=(2, 4),
                fields=["material_id", "formula_pretty", "symmetry"]
            )
    
            print(f"検索結果: {len(docs)}件")
            for doc in docs[:5]:
                print(f"  {doc.material_id}: {doc.formula_pretty}")
    
            return df
    
    # 実行（デモモード）
    API_KEY = os.environ.get('MP_API_KEY', None)
    search_results = search_materials(api_key=API_KEY, demo_mode=(API_KEY is None))
    

## 4\. 結晶構造の変換と操作

### 4.1 超格子の生成

**超格子（supercell）** は、単位格子を整数倍に拡張した構造です。 第一原理計算やドーピング研究で頻繁に使用されます。 

#### コード例5：超格子の生成
    
    
    from pymatgen.core import Structure, Lattice
    import numpy as np
    
    # シリコン構造を作成（前のコード例と同じ）
    a = 5.43
    lattice = Lattice.cubic(a)
    species = ['Si'] * 8
    coords = [
        [0.00, 0.00, 0.00], [0.50, 0.50, 0.00],
        [0.50, 0.00, 0.50], [0.00, 0.50, 0.50],
        [0.25, 0.25, 0.25], [0.75, 0.75, 0.25],
        [0.75, 0.25, 0.75], [0.25, 0.75, 0.75]
    ]
    si_structure = Structure(lattice, species, coords)
    
    print("=== 元の単位格子 ===")
    print(f"化学式: {si_structure.formula}")
    print(f"原子数: {len(si_structure)}")
    print(f"体積: {si_structure.volume:.4f} Å³")
    print(f"格子定数: a = {si_structure.lattice.a:.4f} Å\n")
    
    # 2×2×2 超格子の作成
    supercell_2x2x2 = si_structure.copy()
    supercell_2x2x2.make_supercell([2, 2, 2])
    
    print("=== 2×2×2 超格子 ===")
    print(f"化学式: {supercell_2x2x2.formula}")
    print(f"原子数: {len(supercell_2x2x2)}")
    print(f"体積: {supercell_2x2x2.volume:.4f} Å³")
    print(f"格子定数: a = {supercell_2x2x2.lattice.a:.4f} Å")
    print(f"体積比（超格子/単位格子）: {supercell_2x2x2.volume / si_structure.volume:.1f}\n")
    
    # 3×1×1 超格子（異方的拡張）
    supercell_3x1x1 = si_structure.copy()
    supercell_3x1x1.make_supercell([3, 1, 1])
    
    print("=== 3×1×1 超格子（異方的拡張） ===")
    print(f"原子数: {len(supercell_3x1x1)}")
    print(f"格子定数:")
    print(f"  a = {supercell_3x1x1.lattice.a:.4f} Å")
    print(f"  b = {supercell_3x1x1.lattice.b:.4f} Å")
    print(f"  c = {supercell_3x1x1.lattice.c:.4f} Å\n")
    
    # 変換行列を使った超格子作成
    # 例: 対角化されていない変換（斜め方向への拡張）
    transformation_matrix = [
        [2, 0, 0],
        [0, 2, 0],
        [1, 1, 2]
    ]
    
    supercell_custom = si_structure.copy()
    supercell_custom.make_supercell(transformation_matrix)
    
    print("=== カスタム変換行列による超格子 ===")
    print(f"変換行列:")
    print(np.array(transformation_matrix))
    print(f"\n原子数: {len(supercell_custom)}")
    print(f"体積: {supercell_custom.volume:.4f} Å³")
    
    # 超格子内の特定原子の置換（ドーピングの例）
    print("\n=== ドーピング例: 1つのSiをPに置換 ===")
    doped_structure = supercell_2x2x2.copy()
    doped_structure[0] = 'P'  # 最初のSi原子をPに置換
    
    print(f"置換前の化学式: {supercell_2x2x2.formula}")
    print(f"置換後の化学式: {doped_structure.formula}")
    print(f"ドーピング濃度: 1 / {len(doped_structure)} = "
          f"{1/len(doped_structure)*100:.2f}%")
    

### 4.2 スラブモデルの生成

**スラブモデル** は表面や界面を研究するための構造です。 バルク結晶を特定の結晶面で切断し、真空層を挿入します。 

#### コード例6：スラブモデルの生成（表面科学への応用）
    
    
    from pymatgen.core.surface import SlabGenerator
    from pymatgen.core import Structure, Lattice
    
    # シリコン構造を作成
    a = 5.43
    lattice = Lattice.cubic(a)
    species = ['Si'] * 8
    coords = [
        [0.00, 0.00, 0.00], [0.50, 0.50, 0.00],
        [0.50, 0.00, 0.50], [0.00, 0.50, 0.50],
        [0.25, 0.25, 0.25], [0.75, 0.75, 0.25],
        [0.75, 0.25, 0.75], [0.25, 0.75, 0.75]
    ]
    si_structure = Structure(lattice, species, coords)
    
    print("=== スラブモデルの生成 ===\n")
    
    # (111)面のスラブを生成
    miller_index = (1, 1, 1)
    min_slab_size = 10.0  # スラブの最小厚さ (Å)
    min_vacuum_size = 15.0  # 真空層の最小厚さ (Å)
    
    slabgen = SlabGenerator(
        si_structure,
        miller_index,
        min_slab_size,
        min_vacuum_size,
        center_slab=True  # スラブを真空層の中央に配置
    )
    
    # すべての対称的に異なるスラブを生成
    slabs = slabgen.get_slabs()
    
    print(f"(111)面のスラブモデル: {len(slabs)}種類の終端を検出\n")
    
    for i, slab in enumerate(slabs[:3], 1):  # 最初の3つを表示
        print(f"--- スラブ {i} ---")
        print(f"原子数: {len(slab)}")
        print(f"スラブ厚さ: {slab.get_slab_thickness():.4f} Å")
        print(f"c軸長（真空含む）: {slab.lattice.c:.4f} Å")
        print(f"a, b軸長: {slab.lattice.a:.4f}, {slab.lattice.b:.4f} Å")
    
        # 表面原子と内部原子の区別
        surface_sites = slab.get_surface_sites()
        print(f"表面原子数: {len(surface_sites['top']) + len(surface_sites['bottom'])}")
        print()
    
    # 異なるミラー指数でのスラブ比較
    print("=== 異なる結晶面でのスラブ生成 ===\n")
    
    miller_indices = [(1, 0, 0), (1, 1, 0), (1, 1, 1)]
    
    for hkl in miller_indices:
        slabgen = SlabGenerator(si_structure, hkl, 10.0, 15.0)
        slabs = slabgen.get_slabs()
    
        print(f"({hkl[0]}{hkl[1]}{hkl[2]})面:")
        print(f"  対称的に異なる終端の数: {len(slabs)}")
    
        if len(slabs) > 0:
            first_slab = slabs[0]
            print(f"  代表的なスラブの原子数: {len(first_slab)}")
            print(f"  スラブ厚さ: {first_slab.get_slab_thickness():.4f} Å")
        print()
    
    # スラブをCIFファイルとして保存（例）
    if len(slabs) > 0:
        slab_to_save = slabs[0]
        from pymatgen.io.cif import CifWriter
    
        cif_writer = CifWriter(slab_to_save)
        filename = f"Si_111_slab.cif"
        cif_writer.write_file(filename)
        print(f"スラブモデルを保存しました: {filename}")
    

## 5\. 対称性解析

### 5.1 空間群の同定とWyckoff位置

#### コード例7：対称性解析（空間群、Wyckoff位置）
    
    
    from pymatgen.core import Structure, Lattice
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
    import numpy as np
    
    # いくつかの典型的な結晶構造を定義
    def create_sample_structures():
        """サンプル構造を作成"""
    
        structures = {}
    
        # 1. シリコン（ダイヤモンド構造）
        a = 5.43
        lattice = Lattice.cubic(a)
        species = ['Si'] * 8
        coords = [
            [0.00, 0.00, 0.00], [0.50, 0.50, 0.00],
            [0.50, 0.00, 0.50], [0.00, 0.50, 0.50],
            [0.25, 0.25, 0.25], [0.75, 0.75, 0.25],
            [0.75, 0.25, 0.75], [0.25, 0.75, 0.75]
        ]
        structures['Si (Diamond)'] = Structure(lattice, species, coords)
    
        # 2. NaCl（岩塩構造）
        a = 5.64
        lattice = Lattice.cubic(a)
        species = ['Na', 'Na', 'Na', 'Na', 'Cl', 'Cl', 'Cl', 'Cl']
        coords = [
            [0.0, 0.0, 0.0], [0.5, 0.5, 0.0],
            [0.5, 0.0, 0.5], [0.0, 0.5, 0.5],
            [0.5, 0.0, 0.0], [0.0, 0.5, 0.0],
            [0.0, 0.0, 0.5], [0.5, 0.5, 0.5]
        ]
        structures['NaCl (Rock salt)'] = Structure(lattice, species, coords)
    
        # 3. 鉄（BCC）
        a = 2.87
        lattice = Lattice.cubic(a)
        species = ['Fe', 'Fe']
        coords = [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]
        structures['Fe (BCC)'] = Structure(lattice, species, coords)
    
        return structures
    
    # 構造を作成
    structures = create_sample_structures()
    
    print("=== 対称性解析 ===\n")
    
    for name, structure in structures.items():
        print(f"--- {name} ---")
    
        # SpacegroupAnalyzerの作成
        sga = SpacegroupAnalyzer(structure)
    
        # 基本的な対称性情報
        space_group = sga.get_space_group_symbol()
        space_group_number = sga.get_space_group_number()
        crystal_system = sga.get_crystal_system()
        point_group = sga.get_point_group_symbol()
    
        print(f"空間群: {space_group} (No. {space_group_number})")
        print(f"結晶系: {crystal_system}")
        print(f"点群: {point_group}")
    
        # Wyckoff位置の取得
        symmetrized_structure = sga.get_symmetrized_structure()
    
        print(f"\nWyckoff位置:")
        for i, equiv_sites in enumerate(symmetrized_structure.equivalent_sites):
            species = equiv_sites[0].species_string
            wyckoff = symmetrized_structure.wyckoff_symbols[i]
            multiplicity = len(equiv_sites)
    
            # 代表的な座標（最初の等価サイト）
            representative_coords = equiv_sites[0].frac_coords
    
            print(f"  {species:4s} {wyckoff:3s} (×{multiplicity:2d}): "
                  f"({representative_coords[0]:6.3f}, "
                  f"{representative_coords[1]:6.3f}, "
                  f"{representative_coords[2]:6.3f})")
    
        # 対称操作の数
        symmetry_ops = sga.get_symmetry_operations()
        print(f"\n対称操作の数: {len(symmetry_ops)}")
    
        # 原始格子への変換
        primitive = sga.get_primitive_standard_structure()
        print(f"原始格子の原子数: {len(primitive)} (元の構造: {len(structure)})")
    
        print("\n" + "="*60 + "\n")
    
    # 高度な対称性解析
    print("=== 高度な対称性解析（Si） ===\n")
    
    si_structure = structures['Si (Diamond)']
    sga = SpacegroupAnalyzer(si_structure)
    
    # 対称操作の詳細
    print("対称操作の例（最初の5つ）:")
    symmetry_ops = sga.get_symmetry_operations()
    
    for i, symmop in enumerate(symmetry_ops[:5], 1):
        print(f"\n操作 {i}:")
        print("回転行列:")
        print(symmop.rotation_matrix)
        print(f"並進ベクトル: {symmop.translation_vector}")
    
    # 国際表記とシェーンフリース記号
    print(f"\n国際記号: {sga.get_space_group_symbol()}")
    print(f"点群（国際記号）: {sga.get_point_group_symbol()}")
    
    # 結晶系の詳細情報
    conventional = sga.get_conventional_standard_structure()
    print(f"\n慣用単位格子:")
    print(f"  格子定数: a={conventional.lattice.a:.4f} Å")
    print(f"  原子数: {len(conventional)}")
    

## 6\. 結晶構造の可視化

### 6.1 matplotlibによる2D/3D可視化

#### コード例8：結晶構造の3D可視化
    
    
    from pymatgen.core import Structure, Lattice
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import numpy as np
    
    # NaCl構造を作成（視覚的に分かりやすい）
    a = 5.64
    lattice = Lattice.cubic(a)
    species = ['Na', 'Na', 'Na', 'Na', 'Cl', 'Cl', 'Cl', 'Cl']
    coords = [
        [0.0, 0.0, 0.0], [0.5, 0.5, 0.0],
        [0.5, 0.0, 0.5], [0.0, 0.5, 0.5],
        [0.5, 0.0, 0.0], [0.0, 0.5, 0.0],
        [0.0, 0.0, 0.5], [0.5, 0.5, 0.5]
    ]
    nacl_structure = Structure(lattice, species, coords)
    
    # 2×2×2 超格子を作成（可視化を豊かにする）
    nacl_supercell = nacl_structure.copy()
    nacl_supercell.make_supercell([2, 2, 2])
    
    print("=== 結晶構造の3D可視化 ===\n")
    print(f"構造: NaCl (岩塩構造)")
    print(f"超格子サイズ: 2×2×2")
    print(f"原子数: {len(nacl_supercell)}\n")
    
    # 3Dプロット
    fig = plt.figure(figsize=(14, 6))
    
    # サブプロット1: 球棒モデル
    ax1 = fig.add_subplot(121, projection='3d')
    
    # 元素ごとに色を分ける
    colors = {'Na': 'blue', 'Cl': 'green'}
    sizes = {'Na': 100, 'Cl': 150}
    
    for site in nacl_supercell:
        species = site.species_string
        x, y, z = site.coords
    
        ax1.scatter(x, y, z,
                   c=colors[species],
                   s=sizes[species],
                   alpha=0.6,
                   edgecolors='black',
                   linewidths=0.5)
    
    # 単位格子の枠を描画
    lattice_matrix = nacl_supercell.lattice.matrix
    
    # 単位格子の頂点
    vertices = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 1],
        [1, 1, 1],
        [0, 1, 1]
    ])
    
    # デカルト座標に変換
    vertices_cart = vertices @ lattice_matrix
    
    # 格子の辺を描画
    edges = [
        [0, 1], [1, 2], [2, 3], [3, 0],  # 下面
        [4, 5], [5, 6], [6, 7], [7, 4],  # 上面
        [0, 4], [1, 5], [2, 6], [3, 7]   # 縦の辺
    ]
    
    for edge in edges:
        points = vertices_cart[edge]
        ax1.plot3D(*points.T, 'k-', linewidth=1.5, alpha=0.3)
    
    ax1.set_xlabel('X (Å)', fontweight='bold')
    ax1.set_ylabel('Y (Å)', fontweight='bold')
    ax1.set_zlabel('Z (Å)', fontweight='bold')
    ax1.set_title('NaCl 構造（2×2×2 超格子）', fontweight='bold', fontsize=12)
    
    # 凡例
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w',
               markerfacecolor='blue', markersize=10, label='Na'),
        Line2D([0], [0], marker='o', color='w',
               markerfacecolor='green', markersize=12, label='Cl')
    ]
    ax1.legend(handles=legend_elements, loc='upper right')
    
    # サブプロット2: 単位格子のみ（詳細表示）
    ax2 = fig.add_subplot(122, projection='3d')
    
    for site in nacl_structure:
        species = site.species_string
        x, y, z = site.coords
    
        ax2.scatter(x, y, z,
                   c=colors[species],
                   s=sizes[species]*2,
                   alpha=0.8,
                   edgecolors='black',
                   linewidths=1)
    
        # 原子ラベルを追加
        ax2.text(x, y, z, f'  {species}', fontsize=9)
    
    # 単位格子の枠
    vertices = np.array([
        [0, 0, 0], [a, 0, 0], [a, a, 0], [0, a, 0],
        [0, 0, a], [a, 0, a], [a, a, a], [0, a, a]
    ])
    
    for edge in edges:
        points = vertices[edge]
        ax2.plot3D(*points.T, 'k-', linewidth=2)
    
    ax2.set_xlabel('X (Å)', fontweight='bold')
    ax2.set_ylabel('Y (Å)', fontweight='bold')
    ax2.set_zlabel('Z (Å)', fontweight='bold')
    ax2.set_title('NaCl 単位格子', fontweight='bold', fontsize=12)
    ax2.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    plt.savefig('nacl_structure_3d.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("可視化を保存しました: nacl_structure_3d.png\n")
    
    # ボンドの描画（最近接原子間の線）
    print("=== 最近接原子間距離の可視化 ===")
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 原子を描画
    for site in nacl_structure:
        species = site.species_string
        x, y, z = site.coords
        ax.scatter(x, y, z, c=colors[species], s=sizes[species]*2,
                  alpha=0.8, edgecolors='black', linewidths=1)
    
    # 最近接ボンドを描画
    bond_cutoff = 3.0  # Å（この距離以下をボンドとする）
    
    for i, site1 in enumerate(nacl_structure):
        for j, site2 in enumerate(nacl_structure):
            if i >= j:
                continue
    
            distance = site1.distance(site2)
    
            if distance < bond_cutoff:
                # Na-Cl間のボンドのみ描画（同種原子間は除外）
                if site1.species_string != site2.species_string:
                    x_vals = [site1.coords[0], site2.coords[0]]
                    y_vals = [site1.coords[1], site2.coords[1]]
                    z_vals = [site1.coords[2], site2.coords[2]]
    
                    ax.plot3D(x_vals, y_vals, z_vals,
                             'gray', linewidth=2, alpha=0.5)
    
    # 単位格子の枠
    for edge in edges:
        points = vertices[edge]
        ax.plot3D(*points.T, 'k-', linewidth=2)
    
    ax.set_xlabel('X (Å)', fontweight='bold')
    ax.set_ylabel('Y (Å)', fontweight='bold')
    ax.set_zlabel('Z (Å)', fontweight='bold')
    ax.set_title('NaCl 構造（ボンド表示）', fontweight='bold', fontsize=14)
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    plt.savefig('nacl_structure_bonds.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("ボンド可視化を保存しました: nacl_structure_bonds.png")
    

## 7\. 実践的ワークフロー

### 7.1 完全な解析ワークフロー例

ここまで学んだ技術を統合し、Materials Projectからデータを取得して解析・可視化する実践的なワークフローを示します。 
    
    
    ```mermaid
    flowchart TD
                    A[Materials Projectから構造データ取得] --> B[CIFファイルとして保存]
                    B --> C[対称性解析空間群・Wyckoff位置]
                    C --> D[XRDパターン生成]
                    D --> E[3D構造可視化]
                    E --> F[超格子作成ドーピングシミュレーション]
                    F --> G[研究成果の出力レポート・論文]
    
                    style A fill:#e3f2fd
                    style B fill:#fff3e0
                    style C fill:#e8f5e9
                    style D fill:#fce4ec
                    style E fill:#f3e5f5
                    style F fill:#e0f2f1
                    style G fill:#fff9c4
    ```

#### 実践ワークフローのポイント

  1. **データ取得** : Materials Project APIで信頼性の高い構造データを取得
  2. **データ検証** : 対称性解析で構造の妥当性を確認
  3. **解析・計算** : XRDパターン、電子構造、物性を計算
  4. **可視化** : 直感的な理解のために3D表示やグラフを作成
  5. **変換・応用** : 超格子、ドーピング、表面モデルで応用研究
  6. **自動化** : スクリプト化で大量の材料を効率的に処理

## 8\. 演習問題

### 演習1：pymatgenの基本操作

アルミニウム（Al）のFCC構造を作成し、以下の情報を取得しなさい： 

  * 格子定数 a = 4.05 Å
  * 化学式と原子数
  * 密度（g/cm³）
  * 空間群と点群
  * 最近接原子間距離

また、この構造をCIFファイルとして保存しなさい。

ヒント

FCC構造の原子位置は (0,0,0), (0.5,0.5,0), (0.5,0,0.5), (0,0.5,0.5) です。 `Lattice.cubic()`と`Structure`クラスを使用します。 対称性情報は`SpacegroupAnalyzer`で取得できます。 

### 演習2：超格子とドーピング

シリコン（Si）の2×2×2超格子を作成し、以下の操作を行いなさい： 

  1. 超格子内の原子数を確認する
  2. 1つのSi原子をP（リン）に置換し、ドーピング濃度を計算する
  3. 元の構造と置換後の構造の化学式を比較する
  4. 置換後の構造をCIFファイルとして保存する

ヒント

`make_supercell([2,2,2])`で超格子を作成します。 原子の置換は`structure[index] = 'P'`のように行います。 ドーピング濃度は 置換原子数/全原子数 × 100% で計算します。 

### 演習3：対称性解析

以下の構造について、空間群番号、結晶系、Wyckoff位置を調べなさい： 

  1. 鉄（Fe）のBCC構造（a = 2.87 Å）
  2. 銅（Cu）のFCC構造（a = 3.61 Å）
  3. ペロブスカイト構造 CaTiO₃（a = 3.84 Å） 
     * Ca: (0.5, 0.5, 0.5)
     * Ti: (0, 0, 0)
     * O: (0.5, 0, 0), (0, 0.5, 0), (0, 0, 0.5)

それぞれの構造で対称操作の数も報告しなさい。

ヒント

`SpacegroupAnalyzer`クラスを使用します。 `get_space_group_number()`、`get_crystal_system()`、 `get_symmetrized_structure()`などのメソッドが有用です。 

### 演習4：実践的プロジェクト

以下の手順で、興味のある材料の完全な解析を実施しなさい（APIキーがある場合）： 

  1. Materials Projectから任意の材料（例: GaAs, mp-2534）を取得
  2. 空間群と結晶系を確認
  3. XRDパターンを生成（第4章のコードを参考）
  4. 3D構造を可視化
  5. 2×2×2超格子を作成
  6. すべての結果を1つのPDFレポートにまとめる

APIキーがない場合は、CIFファイルをオンラインデータベース（COD、ICSDなど）から ダウンロードして同様の解析を行いなさい。 

ヒント

matplotlib の `savefig()` で各グラフを画像として保存し、 PyPDF2 や reportlab などのライブラリでPDFにまとめることができます。 あるいは、Jupyter Notebookで実行し、「File → Download as → PDF」で 直接PDFとして保存する方法もあります。 

## まとめと次のステップ

### 本シリーズで習得したスキル

この5章構成の結晶学入門シリーズを通じて、以下を習得しました：

#### 理論的基盤

  * 結晶学の基本概念（単位格子、格子定数、結晶系）
  * 14種類のブラベー格子と230種類の空間群の体系
  * ミラー指数による結晶面・方向の表現
  * X線回折の原理とブラッグの法則
  * 構造因子と消滅則

#### 実践的技術

  * pymatgenライブラリによる結晶構造の操作
  * CIFファイルの読み書きとデータベース連携
  * Materials Project APIを使ったデータ取得
  * 超格子、スラブモデルなどの構造変換
  * 対称性解析と空間群の自動同定
  * 結晶構造の2D/3D可視化
  * XRDパターンのシミュレーションと解析

#### 研究への応用力

  * Materials Informatics研究のための構造記述子の理解
  * 第一原理計算のための構造準備
  * 実験データ（XRD）の解釈と理論計算との対応
  * 材料設計のための構造探索とスクリーニング

### 次に学ぶべき内容

#### 推奨される発展学習

**1\. 材料物性との関係（MS Dojo 他シリーズ）**

  * **材料熱力学入門** ：相図、化学ポテンシャル、安定性解析
  * **材料物性入門** ：バンド構造、状態密度、輸送特性
  * **固体物理入門** ：逆格子、フェルミ面、フォノン

**2\. 計算材料科学（第一原理計算）**

  * 密度汎関数理論（DFT）の基礎
  * VASP、Quantum ESPRESSOなどの第一原理計算ソフトの使い方
  * pymatgenを使った入出力ファイルの自動生成
  * 高スループット計算とワークフロー自動化

**3\. Materials Informatics（MI）**

  * 結晶構造からの記述子（descriptor）抽出
  * 機械学習モデルによる物性予測
  * matminerライブラリの活用
  * ベイズ最適化による材料探索

**4\. 高度な結晶学**

  * 準結晶と非周期構造
  * 薄膜・ナノ材料の結晶学
  * 欠陥構造と転位論
  * 電子顕微鏡との連携（SAED、HREM）

**5\. 材料データベース活用**

  * Materials Project、AFLOW、OQMD等の比較と使い分け
  * 機械可読なデータフォーマット（JSON-LD、HDF5）
  * FAIR原則に基づくデータ管理
  * 自前のデータベース構築

### 学習の継続のために

結晶学とpymatgenの学習を継続するための推奨リソース：

#### 公式ドキュメント・チュートリアル

  * **pymatgen公式ドキュメント** : <https://pymatgen.org/>
  * **Materials Project公式サイト** : <https://next-gen.materialsproject.org/>
  * **pymatgen Jupyter Notebook例** : [GitHub リポジトリ](<https://github.com/materialsproject/pymatgen>)
  * **Materials Project Workshop資料** : 定期的に開催されるワークショップの資料が公開されています

#### 学術論文・教科書

  * **International Tables for Crystallography** (IUCr): 結晶学の決定版リファレンス
  * **"Introduction to Solid State Physics"** by Kittel: 固体物理学の古典的教科書
  * **"Materials Science and Engineering"** by Callister: 材料科学の包括的教科書
  * **pymatgen論文** : Ong et al., Computational Materials Science (2013)

#### オンラインコミュニティ

  * **Materials Project Forum** : 技術的な質問や議論
  * **pymatgen Discourse** : バグ報告や機能リクエスト
  * **Stack Overflow** : プログラミング関連の質問
  * **matsci.org** : 計算材料科学のメーリングリスト

### 最終メッセージ

結晶学は、材料科学の最も基盤的な学問領域の一つです。 原子がどのように配列しているかを理解することは、材料の性質を理解し、 新しい材料を設計する上で不可欠です。 

本シリーズでは、理論的な基礎からpymatgenを使った実践的な計算技術まで、 体系的に学習しました。ここで習得した知識とスキルは、 Materials Informatics、第一原理計算、実験データ解析など、 あらゆる材料研究の基盤となります。 

これからは、興味のある材料や研究テーマに対して、 このシリーズで学んだ技術を積極的に応用してください。 pymatgenとMaterials Projectは強力なツールであり、 あなたの研究を加速させる可能性を秘めています。 

**継続的な学習と実践が、真のスキル習得への道です。** 本シリーズが、あなたの材料科学研究の新たなスタートとなることを願っています。 

### 免責事項

  * 本コンテンツは教育・研究・情報提供のみを目的としており、専門的な助言(法律・会計・技術的保証など)を提供するものではありません。
  * 本コンテンツおよび付随するCode examplesは「現状有姿(AS IS)」で提供され、明示または黙示を問わず、商品性、特定目的適合性、権利非侵害、正確性・完全性、動作・安全性等いかなる保証もしません。
  * 外部リンク、第三者が提供するデータ・ツール・ライブラリ等の内容・可用性・安全性について、作成者および東北大学は一切の責任を負いません。
  * 本コンテンツの利用・実行・解釈により直接的・間接的・付随的・特別・結果的・懲罰的損害が生じた場合でも、適用法で許容される最大限の範囲で、作成者および東北大学は責任を負いません。
  * 本コンテンツの内容は、予告なく変更・更新・提供停止されることがあります。
  * 本コンテンツの著作権・ライセンスは明記された条件(例: CC BY 4.0)に従います。当該ライセンスは通常、無保証条項を含みます。
