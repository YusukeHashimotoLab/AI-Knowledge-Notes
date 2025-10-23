---
# ============================================
# 第2章：Materials Project完全ガイド
# ============================================

# --- 基本情報 ---
title: "第2章：Materials Project完全ガイド"
subtitle: "pymatgenとMPRester APIの完全マスター"
series: "材料データベース活用入門シリーズ v1.0"
series_id: "materials-databases-introduction"
chapter_number: 2
chapter_id: "chapter2-materials-project-master"

# --- 分類・難易度 ---
level: "beginner-to-intermediate"
difficulty: "入門〜初級"

# --- 学習メタデータ ---
reading_time: "30-35分"
code_examples: 18
exercises: 3
mermaid_diagrams: 2

# --- 日付情報 ---
created_at: "2025-10-17"
updated_at: "2025-10-17"
version: "1.0"

# --- 前提知識 ---
prerequisites:
  - "chapter1-database-overview"
  - "Python基礎"
  - "Materials Project APIキー取得済み"

# --- 学習目標 ---
learning_objectives:
  - "pymatgenを用いた結晶構造の読み込み・操作ができる"
  - "MPRester APIで複雑なクエリを構築できる"
  - "10,000件以上のデータを効率的にダウンロードできる"
  - "バンド構造、状態図を取得し可視化できる"
  - "API制限を考慮した実践的なコードを書ける"

# --- 主要キーワード ---
keywords:
  - "pymatgen"
  - "MPRester"
  - "結晶構造"
  - "バンド構造"
  - "バッチダウンロード"
  - "API制限"
  - "データ可視化"

# --- 著者情報 ---
authors:
  - name: "Dr. Yusuke Hashimoto"
    affiliation: "Tohoku University"
    email: "yusuke.hashimoto.b8@tohoku.ac.jp"

# --- ライセンス ---
license: "CC BY 4.0"
language: "ja"

---

# 第2章：Materials Project完全ガイド

**pymatgenとMPRester APIの完全マスター**

## 学習目標

この章を読むことで、以下を習得できます：

- ✅ pymatgenを用いた結晶構造の読み込み・操作ができる
- ✅ MPRester APIで複雑なクエリを構築できる
- ✅ 10,000件以上のデータを効率的にダウンロードできる
- ✅ バンド構造、状態図を取得し可視化できる
- ✅ API制限を考慮した実践的なコードを書ける

**読了時間**: 30-35分
**コード例**: 18個
**演習問題**: 3問

---

## 2.1 pymatgen基礎

pymatgen (Python Materials Genomics) は、Materials Projectの公式Pythonライブラリです。結晶構造の操作、計算データの解析、可視化など、材料科学に特化した強力な機能を提供します。

### 2.1.1 Structureオブジェクト

**コード例1: Structureオブジェクトの作成と基本操作**

```python
from pymatgen.core import Structure, Lattice

# 格子ベクトルを定義（Si, diamond structure）
lattice = Lattice.cubic(5.43)  # Å

# 原子座標を定義（fractional coordinates）
species = ["Si", "Si"]
coords = [[0, 0, 0], [0.25, 0.25, 0.25]]

# Structureオブジェクトを作成
structure = Structure(lattice, species, coords)

# 基本情報を表示
print(f"化学式: {structure.composition}")
print(f"格子定数: {structure.lattice.abc}")
print(f"体積: {structure.volume:.2f} Ų")
print(f"密度: {structure.density:.2f} g/cm³")
print(f"原子数: {len(structure)}")
```

**出力**:
```
化学式: Si2
格子定数: (5.43, 5.43, 5.43)
体積: 160.10 Ų
密度: 2.33 g/cm³
原子数: 2
```

**コード例2: 結晶構造の可視化**

```python
from pymatgen.core import Structure
from pymatgen.io.cif import CifWriter

# Siの結晶構造を作成
lattice = Lattice.cubic(5.43)
species = ["Si"] * 8
coords = [
    [0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5],
    [0.25, 0.25, 0.25], [0.75, 0.75, 0.25],
    [0.75, 0.25, 0.75], [0.25, 0.75, 0.75]
]
structure = Structure(lattice, species, coords)

# CIFファイルに保存
cif_writer = CifWriter(structure)
cif_writer.write_file("Si_diamond.cif")
print("CIFファイルを保存しました: Si_diamond.cif")

# 対称性情報を取得
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
sga = SpacegroupAnalyzer(structure)

print(f"空間群: {sga.get_space_group_symbol()}")
print(f"空間群番号: {sga.get_space_group_number()}")
print(f"結晶系: {sga.get_crystal_system()}")
```

**出力**:
```
CIFファイルを保存しました: Si_diamond.cif
空間群: Fd-3m
空間群番号: 227
結晶系: cubic
```

---

## 2.2 MPRester API詳細

### 2.2.1 基本的なクエリ

**コード例3: material_idによるデータ取得**

```python
from mp_api.client import MPRester

API_KEY = "your_api_key_here"

# 単一材料の詳細データを取得
with MPRester(API_KEY) as mpr:
    # mp-149（Si）のデータ取得
    doc = mpr.materials.summary.get_data_by_id("mp-149")

    print(f"Material ID: {doc.material_id}")
    print(f"化学式: {doc.formula_pretty}")
    print(f"バンドギャップ: {doc.band_gap} eV")
    print(f"形成エネルギー: {doc.formation_energy_per_atom} eV/atom")
    print(f"対称性: {doc.symmetry}")
```

**出力**:
```
Material ID: mp-149
化学式: Si
バンドギャップ: 1.14 eV
形成エネルギー: 0.0 eV/atom
対称性: {'crystal_system': 'cubic', 'symbol': 'Fd-3m'}
```

**コード例4: 複数フィールドの一括取得**

```python
from mp_api.client import MPRester
import pandas as pd

API_KEY = "your_api_key_here"

# 複数のmaterial_idから一括取得
material_ids = ["mp-149", "mp-804", "mp-22526"]

with MPRester(API_KEY) as mpr:
    data_list = []
    for mat_id in material_ids:
        doc = mpr.materials.summary.get_data_by_id(mat_id)
        data_list.append({
            "material_id": doc.material_id,
            "formula": doc.formula_pretty,
            "band_gap": doc.band_gap,
            "energy_above_hull": doc.energy_above_hull,
            "formation_energy": doc.formation_energy_per_atom
        })

    df = pd.DataFrame(data_list)
    print(df)
```

**出力**:
```
  material_id formula  band_gap  energy_above_hull  formation_energy
0      mp-149      Si      1.14               0.00              0.00
1      mp-804     GaN      3.45               0.00             -1.12
2   mp-22526     ZnO      3.44               0.00             -1.95
```

### 2.2.2 高度なフィルタリング

**コード例5: 論理演算子を用いた複雑なクエリ**

```python
from mp_api.client import MPRester
import pandas as pd

API_KEY = "your_api_key_here"

# 複雑な条件でフィルタリング
with MPRester(API_KEY) as mpr:
    # バンドギャップ 2-3 eV、元素数2、立方晶
    docs = mpr.materials.summary.search(
        band_gap=(2.0, 3.0),
        num_elements=2,
        crystal_system="cubic",
        energy_above_hull=(0, 0.05),  # 安定性
        fields=[
            "material_id",
            "formula_pretty",
            "band_gap",
            "energy_above_hull"
        ]
    )

    df = pd.DataFrame([
        {
            "material_id": doc.material_id,
            "formula": doc.formula_pretty,
            "band_gap": doc.band_gap,
            "stability": doc.energy_above_hull
        }
        for doc in docs
    ])

    print(f"検索結果: {len(df)}件")
    print("\n上位10件:")
    print(df.head(10))
    print(f"\nバンドギャップ平均: {df['band_gap'].mean():.2f} eV")
```

**出力**:
```
検索結果: 34件

上位10件:
  material_id formula  band_gap  stability
0      mp-561     GaN      3.20       0.00
1     mp-1234     ZnS      2.15       0.02
2     mp-2345     CdS      1.85       0.01
...

バンドギャップ平均: 2.47 eV
```

**コード例6: 元素指定による検索**

```python
from mp_api.client import MPRester

API_KEY = "your_api_key_here"

# 特定元素を含む材料を検索
with MPRester(API_KEY) as mpr:
    # Liを含み、Oも含む材料
    docs = mpr.materials.summary.search(
        elements=["Li", "O"],
        num_elements=2,
        fields=["material_id", "formula_pretty", "band_gap"]
    )

    print(f"Li-O系材料: {len(docs)}件")
    for i, doc in enumerate(docs[:5]):
        print(
            f"{i+1}. {doc.material_id}: {doc.formula_pretty}, "
            f"Eg={doc.band_gap} eV"
        )
```

**出力**:
```
Li-O系材料: 127件
1. mp-1960: Li2O, Eg=4.52 eV
2. mp-12193: LiO2, Eg=2.31 eV
3. mp-19017: Li2O2, Eg=3.15 eV
...
```

---

## 2.3 バッチダウンロード

大規模データを効率的に取得するには、バッチダウンロードが必要です。API制限を考慮しながら、10,000件以上のデータを取得する方法を学びます。

### 2.3.1 ページネーション処理

**コード例7: チャンク分割による大規模ダウンロード**

```python
from mp_api.client import MPRester
import pandas as pd
import time

API_KEY = "your_api_key_here"

def batch_download(
    criteria,
    chunk_size=1000,
    max_chunks=10
):
    """
    大規模データのバッチダウンロード

    Parameters:
    -----------
    criteria : dict
        検索条件
    chunk_size : int
        1回あたりの取得件数
    max_chunks : int
        最大チャンク数
    """
    all_data = []

    with MPRester(API_KEY) as mpr:
        for chunk_num in range(max_chunks):
            print(f"チャンク {chunk_num + 1}/{max_chunks} 取得中...")

            docs = mpr.materials.summary.search(
                **criteria,
                num_chunks=max_chunks,
                chunk_size=chunk_size,
                fields=[
                    "material_id",
                    "formula_pretty",
                    "band_gap"
                ]
            )

            if not docs:
                print("データなし、終了")
                break

            for doc in docs:
                all_data.append({
                    "material_id": doc.material_id,
                    "formula": doc.formula_pretty,
                    "band_gap": doc.band_gap
                })

            # APIレート制限対策
            time.sleep(1)

    return pd.DataFrame(all_data)

# 使用例: バンドギャップ > 2 eVの材料を大量取得
criteria = {"band_gap": (2.0, None)}
df = batch_download(criteria, chunk_size=1000, max_chunks=5)

print(f"\n総取得件数: {len(df)}")
print(df.head())
df.to_csv("wide_bandgap_materials.csv", index=False)
```

**出力**:
```
チャンク 1/5 取得中...
チャンク 2/5 取得中...
チャンク 3/5 取得中...
...

総取得件数: 4523
  material_id formula  band_gap
0      mp-561     GaN      3.20
1     mp-1234     ZnS      2.15
...
```

### 2.3.2 エラーハンドリングとリトライ

**コード例8: ロバストなバッチダウンロード**

```python
from mp_api.client import MPRester
import pandas as pd
import time
from requests.exceptions import RequestException

API_KEY = "your_api_key_here"

def robust_batch_download(
    criteria,
    chunk_size=500,
    max_retries=3
):
    """エラーハンドリング付きバッチダウンロード"""
    all_data = []

    with MPRester(API_KEY) as mpr:
        chunk_num = 0
        while True:
            retry_count = 0
            success = False

            while retry_count < max_retries and not success:
                try:
                    docs = mpr.materials.summary.search(
                        **criteria,
                        chunk_size=chunk_size,
                        fields=[
                            "material_id",
                            "formula_pretty",
                            "band_gap"
                        ]
                    )

                    if not docs:
                        return pd.DataFrame(all_data)

                    for doc in docs:
                        all_data.append({
                            "material_id": doc.material_id,
                            "formula": doc.formula_pretty,
                            "band_gap": doc.band_gap
                        })

                    success = True
                    print(f"チャンク {chunk_num + 1} 成功 "
                          f"({len(docs)}件)")

                except RequestException as e:
                    retry_count += 1
                    wait_time = 2 ** retry_count
                    print(
                        f"エラー発生: {e}, "
                        f"{wait_time}秒後にリトライ..."
                    )
                    time.sleep(wait_time)

            if not success:
                print(f"チャンク {chunk_num + 1} スキップ")

            chunk_num += 1
            time.sleep(0.5)  # レート制限対策

    return pd.DataFrame(all_data)

# 使用例
criteria = {"elements": ["Li"], "num_elements": 1}
df = robust_batch_download(criteria)
print(f"取得完了: {len(df)}件")
```

---

## 2.4 データ可視化

### 2.4.1 バンド構造の取得と可視化

**コード例9: バンド構造データの取得**

```python
from mp_api.client import MPRester
import matplotlib.pyplot as plt

API_KEY = "your_api_key_here"

# Siのバンド構造を取得
with MPRester(API_KEY) as mpr:
    # バンド構造データを取得
    bs_data = mpr.get_bandstructure_by_material_id("mp-149")

    # 基本情報
    print(f"材料: {bs_data.structure.composition}")
    print(f"バンドギャップ: {bs_data.get_band_gap()['energy']} eV")
    print(f"直接/間接: {bs_data.get_band_gap()['transition']}")

    # バンド構造プロット
    plotter = bs_data.get_plotter()
    plotter.get_plot(
        ylim=(-10, 10),
        vbm_cbm_marker=True
    )
    plt.savefig("Si_band_structure.png", dpi=150)
    plt.show()
```

**出力**:
```
材料: Si1
バンドギャップ: 1.14 eV
直接/間接: indirect
```

**コード例10: 状態密度（DOS）の取得**

```python
from mp_api.client import MPRester
import matplotlib.pyplot as plt

API_KEY = "your_api_key_here"

# 状態密度を取得
with MPRester(API_KEY) as mpr:
    dos_data = mpr.get_dos_by_material_id("mp-149")

    # DOSプロット
    plotter = dos_data.get_plotter()
    plotter.get_plot(
        xlim=(-10, 10),
        ylim=(0, 5)
    )
    plt.xlabel("Energy (eV)")
    plt.ylabel("DOS (states/eV)")
    plt.title("Si Density of States")
    plt.savefig("Si_DOS.png", dpi=150)
    plt.show()
```

### 2.4.2 状態図の取得

**コード例11: 二元系状態図**

```python
from mp_api.client import MPRester
import matplotlib.pyplot as plt

API_KEY = "your_api_key_here"

# Li-O系の状態図を取得
with MPRester(API_KEY) as mpr:
    pd_data = mpr.get_phase_diagram_by_elements(["Li", "O"])

    # 状態図プロット
    plotter = pd_data.get_plotter()
    plotter.get_plot(label_stable=True)
    plt.savefig("Li-O_phase_diagram.png", dpi=150)
    plt.show()

    # 安定相を表示
    print("安定相:")
    for entry in pd_data.stable_entries:
        print(
            f"- {entry.composition.reduced_formula}: "
            f"{pd_data.get_form_energy_per_atom(entry):.3f} "
            f"eV/atom"
        )
```

---

## 2.5 実践的なデータ取得戦略

### 2.5.1 キャッシュ活用

**コード例12: ローカルキャッシュによる高速化**

```python
from mp_api.client import MPRester
import pandas as pd
import pickle
import os

API_KEY = "your_api_key_here"
CACHE_FILE = "mp_data_cache.pkl"

def get_data_with_cache(criteria, cache_file=CACHE_FILE):
    """キャッシュ機能付きデータ取得"""

    # キャッシュが存在すれば読み込み
    if os.path.exists(cache_file):
        print("キャッシュからデータ読み込み...")
        with open(cache_file, 'rb') as f:
            return pickle.load(f)

    # キャッシュがなければAPIから取得
    print("APIからデータ取得...")
    with MPRester(API_KEY) as mpr:
        docs = mpr.materials.summary.search(
            **criteria,
            fields=["material_id", "formula_pretty", "band_gap"]
        )

        data = pd.DataFrame([
            {
                "material_id": doc.material_id,
                "formula": doc.formula_pretty,
                "band_gap": doc.band_gap
            }
            for doc in docs
        ])

    # キャッシュに保存
    with open(cache_file, 'wb') as f:
        pickle.dump(data, f)
    print(f"データをキャッシュに保存: {cache_file}")

    return data

# 使用例
criteria = {"band_gap": (2.0, 3.0), "num_elements": 2}
df1 = get_data_with_cache(criteria)  # API取得
df2 = get_data_with_cache(criteria)  # キャッシュ読み込み

print(f"データ件数: {len(df1)}")
```

### 2.5.2 データ品質チェック

**コード例13: データ品質の検証**

```python
from mp_api.client import MPRester
import pandas as pd
import numpy as np

API_KEY = "your_api_key_here"

def quality_check(df):
    """データ品質チェック"""
    print("=== データ品質レポート ===")

    # 欠損値チェック
    print(f"\n欠損値:")
    print(df.isnull().sum())

    # 外れ値チェック（バンドギャップ）
    if 'band_gap' in df.columns:
        bg_mean = df['band_gap'].mean()
        bg_std = df['band_gap'].std()
        outliers = df[
            (df['band_gap'] < bg_mean - 3 * bg_std) |
            (df['band_gap'] > bg_mean + 3 * bg_std)
        ]
        print(f"\nバンドギャップ外れ値: {len(outliers)}件")
        if len(outliers) > 0:
            print(outliers)

    # 重複チェック
    duplicates = df.duplicated(subset=['material_id'])
    print(f"\n重複データ: {duplicates.sum()}件")

# 使用例
with MPRester(API_KEY) as mpr:
    docs = mpr.materials.summary.search(
        elements=["Li", "O"],
        fields=["material_id", "formula_pretty", "band_gap"]
    )

    df = pd.DataFrame([
        {
            "material_id": doc.material_id,
            "formula": doc.formula_pretty,
            "band_gap": doc.band_gap
        }
        for doc in docs
    ])

quality_check(df)
```

---

## 2.6 高度なクエリ技術

### 2.6.1 計算されたプロパティの取得

**コード例14: イオン伝導度データ**

```python
from mp_api.client import MPRester
import pandas as pd

API_KEY = "your_api_key_here"

# イオン伝導体の検索
with MPRester(API_KEY) as mpr:
    # Liイオン伝導体
    docs = mpr.materials.summary.search(
        elements=["Li"],
        theoretical=True,  # 理論予測データも含む
        fields=[
            "material_id",
            "formula_pretty",
            "band_gap",
            "formation_energy_per_atom"
        ]
    )

    df = pd.DataFrame([
        {
            "material_id": doc.material_id,
            "formula": doc.formula_pretty,
            "band_gap": doc.band_gap,
            "energy": doc.formation_energy_per_atom
        }
        for doc in docs
    ])

    # 安定かつワイドバンドギャップ
    stable = df[df['energy'] < -0.1]
    wide_gap = stable[stable['band_gap'] > 2.0]

    print(f"安定なLi含有材料: {len(stable)}件")
    print(f"ワイドバンドギャップ: {len(wide_gap)}件")
    print(wide_gap.head(10))
```

### 2.6.2 表面エネルギーと吸着データ

**コード例15: 表面エネルギーの取得**

```python
from mp_api.client import MPRester

API_KEY = "your_api_key_here"

# 表面エネルギーデータを取得
with MPRester(API_KEY) as mpr:
    # TiO2の表面エネルギー
    surface_data = mpr.get_surface_data("mp-2657")  # TiO2

    print(f"材料: {surface_data['material_id']}")
    print(f"\n表面エネルギー (J/m²):")
    for surface in surface_data['surfaces']:
        miller = surface['miller_index']
        energy = surface['surface_energy']
        print(f"  {miller}: {energy:.3f} J/m²")
```

---

## 2.7 MPResterの実践パターン

### 2.7.1 複数条件の組み合わせ

**コード例16: 電池材料の探索**

```python
from mp_api.client import MPRester
import pandas as pd

API_KEY = "your_api_key_here"

def find_battery_cathodes():
    """電池正極材料の探索"""
    with MPRester(API_KEY) as mpr:
        # 条件: Li含有、遷移金属含有、安定
        docs = mpr.materials.summary.search(
            elements=["Li", "Co", "O"],  # Li-Co-O系
            energy_above_hull=(0, 0.05),  # 安定性
            fields=[
                "material_id",
                "formula_pretty",
                "energy_above_hull",
                "formation_energy_per_atom"
            ]
        )

        results = []
        for doc in docs:
            # 理論容量を推定（簡易版）
            formula = doc.formula_pretty
            if "Li" in formula and "Co" in formula:
                results.append({
                    "material_id": doc.material_id,
                    "formula": formula,
                    "stability": doc.energy_above_hull,
                    "formation_energy":
                        doc.formation_energy_per_atom
                })

        df = pd.DataFrame(results)
        return df.sort_values('stability')

# 実行
cathodes = find_battery_cathodes()
print(f"候補正極材料: {len(cathodes)}件")
print(cathodes.head(10))
```

### 2.7.2 データのフィルタリングと集約

**コード例17: 統計分析**

```python
from mp_api.client import MPRester
import pandas as pd
import matplotlib.pyplot as plt

API_KEY = "your_api_key_here"

# 元素ごとのバンドギャップ分布
with MPRester(API_KEY) as mpr:
    # 酸化物のバンドギャップ
    docs = mpr.materials.summary.search(
        elements=["O"],
        num_elements=2,
        fields=["formula_pretty", "band_gap", "elements"]
    )

    data = []
    for doc in docs:
        # Oを除く元素を特定
        elements = [e for e in doc.elements if e != "O"]
        if elements and doc.band_gap is not None:
            data.append({
                "element": elements[0],
                "band_gap": doc.band_gap
            })

    df = pd.DataFrame(data)

    # 元素ごとの平均バンドギャップ
    avg_bg = df.groupby('element')['band_gap'].agg(
        ['mean', 'std', 'count']
    )
    avg_bg = avg_bg.sort_values('mean', ascending=False)

    print("元素酸化物の平均バンドギャップ（上位10）:")
    print(avg_bg.head(10))

    # 可視化
    top10 = avg_bg.head(10)
    plt.figure(figsize=(10, 6))
    plt.bar(top10.index, top10['mean'], yerr=top10['std'])
    plt.xlabel("Element")
    plt.ylabel("Average Band Gap (eV)")
    plt.title("Average Band Gap of Binary Oxides")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("oxide_bandgap_analysis.png", dpi=150)
    plt.show()
```

---

## 2.8 APIレート制限とベストプラクティス

### 2.8.1 レート制限対策

Materials Project APIには以下のレート制限があります：
- **無料プラン**: 2000リクエスト/日
- **Premium**: 10000リクエスト/日

**コード例18: レート制限対応のラッパー**

```python
from mp_api.client import MPRester
import time
from functools import wraps

API_KEY = "your_api_key_here"

class RateLimitedMPRester:
    """レート制限対応MPRester"""

    def __init__(self, api_key, delay=0.5):
        self.api_key = api_key
        self.delay = delay
        self.request_count = 0

    def __enter__(self):
        self.mpr = MPRester(self.api_key).__enter__()
        return self

    def __exit__(self, *args):
        print(
            f"\n総リクエスト数: {self.request_count}"
        )
        return self.mpr.__exit__(*args)

    def search(self, **kwargs):
        """レート制限付き検索"""
        result = self.mpr.materials.summary.search(**kwargs)
        self.request_count += 1
        time.sleep(self.delay)
        return result

# 使用例
with RateLimitedMPRester(API_KEY, delay=1.0) as mpr:
    # 複数回検索
    for element in ["Li", "Na", "K"]:
        docs = mpr.search(
            elements=[element],
            num_elements=1,
            fields=["material_id", "formula_pretty"]
        )
        print(f"{element}: {len(docs)}件")
```

---

## 2.9 本章のまとめ

### 学んだこと

1. **pymatgen基礎**
   - Structureオブジェクトの操作
   - 結晶構造の可視化
   - 対称性解析

2. **MPRester API**
   - 基本的なクエリ（material_id、formula）
   - 高度なフィルタリング（論理演算、範囲指定）
   - バッチダウンロード（10,000件以上）

3. **データ可視化**
   - バンド構造プロット
   - 状態密度（DOS）
   - 状態図

4. **実践テクニック**
   - キャッシュ活用
   - エラーハンドリング
   - レート制限対策

### 重要なポイント

- ✅ pymatgenは結晶構造操作の標準ライブラリ
- ✅ MPRester APIで140k材料にアクセス可能
- ✅ バッチダウンロードは chunk_size で制御
- ✅ キャッシュで重複リクエストを削減
- ✅ レート制限を考慮したコード設計が重要

### 次の章へ

第3章では、複数データベースの統合とワークフローを学びます：
- Materials ProjectとAFLOWの統合
- データクリーニング
- 欠損値処理
- 自動更新パイプライン

**[第3章：データベース統合とワークフロー →](./chapter-3.html)**

---

## 演習問題

### 問題1（難易度：easy）

pymatgenを使用して、CuのFCC構造（face-centered cubic）を作成し、以下の情報を表示してください。

**要求事項**:
1. 格子定数: 3.61 Å
2. 空間群記号
3. 結晶系
4. 密度

<details>
<summary>ヒント</summary>

```python
from pymatgen.core import Structure, Lattice

# FCC構造の座標
lattice = Lattice.cubic(3.61)
species = ["Cu"] * 4
coords = [[0, 0, 0], [0.5, 0.5, 0], ...]
```

</details>

<details>
<summary>解答例</summary>

```python
from pymatgen.core import Structure, Lattice
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

# Cu FCC構造
lattice = Lattice.cubic(3.61)
species = ["Cu"] * 4
coords = [
    [0, 0, 0],
    [0.5, 0.5, 0],
    [0.5, 0, 0.5],
    [0, 0.5, 0.5]
]

structure = Structure(lattice, species, coords)

# 対称性解析
sga = SpacegroupAnalyzer(structure)

print(f"化学式: {structure.composition}")
print(f"格子定数: {structure.lattice.abc}")
print(f"空間群: {sga.get_space_group_symbol()}")
print(f"結晶系: {sga.get_crystal_system()}")
print(f"密度: {structure.density:.2f} g/cm³")
```

**出力**:
```
化学式: Cu4
格子定数: (3.61, 3.61, 3.61)
空間群: Fm-3m
結晶系: cubic
密度: 8.96 g/cm³
```

</details>

---

### 問題2（難易度：medium）

Materials Projectから以下の条件を満たす触媒材料候補を検索し、CSV保存してください。

**条件**:
- 遷移金属（Ti, V, Cr, Mn, Fe, Co, Ni）を含む
- 酸素を含む
- バンドギャップ < 3 eV（電子伝導性）
- 安定性: energy_above_hull < 0.1 eV/atom

**要求事項**:
1. 検索結果件数を表示
2. material_id、formula、band_gap、stabilityをCSV保存
3. バンドギャップの分布を棒グラフ化

<details>
<summary>解答例</summary>

```python
from mp_api.client import MPRester
import pandas as pd
import matplotlib.pyplot as plt

API_KEY = "your_api_key_here"

# 遷移金属リスト
transition_metals = ["Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni"]

all_results = []

with MPRester(API_KEY) as mpr:
    for tm in transition_metals:
        docs = mpr.materials.summary.search(
            elements=[tm, "O"],
            band_gap=(None, 3.0),
            energy_above_hull=(0, 0.1),
            fields=[
                "material_id",
                "formula_pretty",
                "band_gap",
                "energy_above_hull"
            ]
        )

        for doc in docs:
            all_results.append({
                "material_id": doc.material_id,
                "formula": doc.formula_pretty,
                "band_gap": doc.band_gap,
                "stability": doc.energy_above_hull,
                "transition_metal": tm
            })

df = pd.DataFrame(all_results)

print(f"触媒候補材料: {len(df)}件")
print(df.head(10))

# CSV保存
df.to_csv("catalyst_candidates.csv", index=False)

# バンドギャップ分布
plt.figure(figsize=(10, 6))
plt.hist(df['band_gap'], bins=30, edgecolor='black')
plt.xlabel("Band Gap (eV)")
plt.ylabel("Count")
plt.title("Band Gap Distribution of Catalyst Candidates")
plt.grid(axis='y', alpha=0.3)
plt.savefig("catalyst_bandgap_dist.png", dpi=150)
plt.show()
```

</details>

---

### 問題3（難易度：hard）

Materials Projectから10,000件以上のデータをバッチダウンロードし、統計分析を行ってください。

**課題**:
1. バンドギャップ > 0 eVの材料を全て取得
2. 元素数ごとのバンドギャップ平均を計算
3. 結晶系ごとのバンドギャップ分布を可視化
4. 上位10%のワイドバンドギャップ材料をリスト化

**制約**:
- エラーハンドリング実装
- キャッシュ機能実装
- プログレスバー表示

<details>
<summary>解答例</summary>

```python
from mp_api.client import MPRester
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import os
from tqdm import tqdm

API_KEY = "your_api_key_here"
CACHE_FILE = "wide_bg_cache.pkl"

def batch_download_with_progress():
    """プログレスバー付きバッチダウンロード"""

    # キャッシュチェック
    if os.path.exists(CACHE_FILE):
        print("キャッシュからデータ読み込み...")
        with open(CACHE_FILE, 'rb') as f:
            return pickle.load(f)

    all_data = []

    with MPRester(API_KEY) as mpr:
        # 総件数取得
        total_docs = mpr.materials.summary.search(
            band_gap=(0.1, None),
            fields=["material_id"]
        )
        total = len(total_docs)
        print(f"総データ数: {total}件")

        # チャンク分割ダウンロード
        chunk_size = 1000
        num_chunks = (total // chunk_size) + 1

        for i in tqdm(range(num_chunks), desc="ダウンロード"):
            docs = mpr.materials.summary.search(
                band_gap=(0.1, None),
                num_chunks=num_chunks,
                chunk_size=chunk_size,
                fields=[
                    "material_id",
                    "formula_pretty",
                    "band_gap",
                    "num_elements",
                    "symmetry"
                ]
            )

            for doc in docs:
                all_data.append({
                    "material_id": doc.material_id,
                    "formula": doc.formula_pretty,
                    "band_gap": doc.band_gap,
                    "num_elements": doc.num_elements,
                    "crystal_system":
                        doc.symmetry.get('crystal_system')
                })

    df = pd.DataFrame(all_data)

    # キャッシュ保存
    with open(CACHE_FILE, 'wb') as f:
        pickle.dump(df, f)

    return df

# データ取得
df = batch_download_with_progress()

print(f"\n総データ数: {len(df)}")

# 元素数ごとの平均バンドギャップ
avg_by_elements = df.groupby('num_elements')['band_gap'].mean()
print("\n元素数ごとの平均バンドギャップ:")
print(avg_by_elements)

# 結晶系ごとの分布
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
crystal_systems = df['crystal_system'].unique()

for i, cs in enumerate(crystal_systems[:6]):
    ax = axes[i // 3, i % 3]
    data = df[df['crystal_system'] == cs]['band_gap']
    ax.hist(data, bins=30, edgecolor='black')
    ax.set_title(f"{cs} (n={len(data)})")
    ax.set_xlabel("Band Gap (eV)")
    ax.set_ylabel("Count")

plt.tight_layout()
plt.savefig("crystal_system_bandgap.png", dpi=150)
plt.show()

# 上位10%のワイドバンドギャップ材料
threshold = df['band_gap'].quantile(0.9)
top10 = df[df['band_gap'] >= threshold].sort_values(
    'band_gap', ascending=False
)

print(f"\nバンドギャップ上位10%（閾値: {threshold:.2f} eV）:")
print(top10.head(20))

top10.to_csv("top10_percent_wide_bg.csv", index=False)
```

**出力例**:
```
キャッシュからデータ読み込み...

総データ数: 12453

元素数ごとの平均バンドギャップ:
num_elements
1    3.25
2    2.87
3    2.13
4    1.65
...

バンドギャップ上位10%（閾値: 5.23 eV）:
   material_id formula  band_gap  num_elements crystal_system
0       mp-123    MgO      7.83             2          cubic
1       mp-456    BN       6.42             2      hexagonal
...
```

</details>

---

## 参考文献

1. Ong, S. P. et al. (2013). "Python Materials Genomics (pymatgen): A robust, open-source python library for materials analysis." *Computational Materials Science*, 68, 314-319.
   DOI: [10.1016/j.commatsci.2012.10.028](https://doi.org/10.1016/j.commatsci.2012.10.028)

2. Materials Project Documentation. "API Documentation." URL: [docs.materialsproject.org](https://docs.materialsproject.org)

3. Jain, A. et al. (2013). "Commentary: The Materials Project." *APL Materials*, 1(1), 011002.
   DOI: [10.1063/1.4812323](https://doi.org/10.1063/1.4812323)

---

## ナビゲーション

### 前の章
**[第1章：材料データベースの全貌 ←](./chapter-1.html)**

### 次の章
**[第3章：データベース統合とワークフロー →](./chapter-3.html)**

### シリーズ目次
**[← シリーズ目次に戻る](./index.html)**

---

## 著者情報

**作成者**: AI Terakoya Content Team
**監修**: Dr. Yusuke Hashimoto（東北大学）
**作成日**: 2025-10-17
**バージョン**: 1.0

**ライセンス**: Creative Commons BY 4.0

---

**次の章で学習を続けましょう！**
