# 第3章：実装：材料データベースの活用

**材料データベースを使って、実際のデータで材料分類を学ぶ**

---

## 3.1 材料データベースとは

材料科学の研究において、膨大な材料データを効率的に管理・活用することは非常に重要です。材料データベースは、世界中の研究者が蓄積してきた実験データや計算データを一元管理し、新しい材料開発を加速するための強力なツールです。

### 3.1.1 主要な材料データベース

現在、研究者が利用できる主要な材料データベースには以下のようなものがあります：

| データベース | 提供機関 | 材料数 | 特徴 |
|------------|---------|--------|------|
| **Materials Project** | Lawrence Berkeley National Lab | 140,000+ | 第一原理計算データ、API提供 |
| **AFLOW** | Duke University | 3,500,000+ | 結晶構造データベース |
| **NIMS MatNavi** | 物質・材料研究機構 | 多数 | 日本発の材料データベース |
| **ICSD** | FIZ Karlsruhe | 250,000+ | 無機結晶構造データベース |

本章では、最も広く使われている**Materials Project**を例に、材料データベースの活用方法を学びます。

### 3.1.2 Materials Projectの概要

Materials Projectは、密度汎関数理論（DFT）に基づく第一原理計算により、材料の物性データを計算・提供しているオープンデータベースです。

**主な提供データ**:
- バンドギャップ（Band Gap）
- 生成エネルギー（Formation Energy）
- 結晶構造（Crystal Structure）
- 密度（Density）
- 弾性定数（Elastic Constants）
- 圧電特性（Piezoelectric Properties）

**アクセス方法**:
1. **Webインターフェース**: https://materialsproject.org/
2. **Python API**: `mp-api`ライブラリを使用
3. **REST API**: プログラムから直接アクセス

**💡 Pro Tip:**
Materials Projectを利用するには、無料のアカウント登録が必要です。登録後、APIキーを取得することで、Pythonスクリプトから直接データにアクセスできます。

---

## 3.2 実装手順：材料データの取得

材料データベースからデータを取得し、活用するための基本的な手順を学びましょう。

### 3.2.1 環境準備

Materials Project APIを使用するために、必要なライブラリをインストールします。

```bash
# 必要なライブラリのインストール
pip install mp-api
pip install matplotlib numpy
```

**必要なライブラリ**:
- `mp-api`: Materials Project公式APIクライアント
- `matplotlib`: データ可視化
- `numpy`: 数値計算

### 3.2.2 APIキーの設定

Materials Projectのウェブサイトでアカウントを作成し、APIキーを取得します。

```python
# APIキーの設定方法（環境変数を使用）
import os
os.environ["MP_API_KEY"] = "your_api_key_here"
```

⚠️ **注意**: APIキーは秘密情報です。コードに直接書き込まず、環境変数や設定ファイルを使用してください。

---

## 3.3 コード例1：材料データの取得

それでは、実際にPythonコードを使って材料データを取得する方法を見ていきましょう。

### 3.3.1 基本的なデータ取得

以下のコードは、Materials Project APIをシミュレートして、材料データの取得方法を示しています。

```python
# ===================================
# Example 1: 材料データベースからのデータ取得
# ===================================

import json
from typing import Dict, List, Optional

# サンプル材料データ（実際のMaterials Projectの構造を模擬）
SAMPLE_MATERIALS_DATA = [
    {
        "material_id": "mp-66",
        "formula": "Si",
        "material_type": "半導体",
        "band_gap": 1.14,
        "formation_energy": -5.424,
        "density": 2.33,
        "structure_type": "diamond",
        "space_group": "Fd-3m"
    },
    # ... その他の材料データ
]

def get_material_by_formula(formula: str) -> Optional[Dict]:
    """
    化学式を指定して材料データを取得

    Args:
        formula (str): 化学式（例: "Si", "Al2O3"）

    Returns:
        Optional[Dict]: 材料データの辞書、見つからない場合はNone
    """
    for material in SAMPLE_MATERIALS_DATA:
        if material["formula"] == formula:
            return material
    return None
```

### 3.3.2 実行結果

上記のコードを実行すると、以下のような結果が得られます。

```
【1】化学式による検索
==================================================
Material ID: mp-66
化学式: Si
材料分類: 半導体
==================================================
バンドギャップ: 1.14 eV
生成エネルギー: -5.424 eV/atom
密度: 2.33 g/cm³
結晶構造: diamond
空間群: Fd-3m
==================================================
```

### 3.3.3 材料分類による検索

材料を分類（金属、半導体、セラミックス）ごとに検索する関数も実装できます。

```python
def get_materials_by_type(material_type: str) -> List[Dict]:
    """
    材料分類で材料データを検索

    Args:
        material_type (str): 材料分類（"金属", "セラミックス", "半導体"）

    Returns:
        List[Dict]: 該当する材料データのリスト
    """
    filtered_materials = [
        material for material in SAMPLE_MATERIALS_DATA
        if material["material_type"] == material_type
    ]
    return filtered_materials
```

**実行結果**:
```
【2】材料分類による検索
金属材料: 2件
  - Al: fcc structure
  - Fe: bcc structure

セラミックス材料: 2件
  - Al2O3: バンドギャップ 6.28 eV
  - SiO2: バンドギャップ 5.61 eV
```

### 3.3.4 データの永続化

取得したデータをJSONファイルに保存することで、オフラインでも利用できるようにします。

```python
# JSONファイルへの保存
output_file = "output/chapter3/materials_data.json"

with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(SAMPLE_MATERIALS_DATA, f, ensure_ascii=False, indent=2)

print(f"材料データを保存しました: {output_file}")
```

**🎯 ベストプラクティス**:
- UTF-8エンコーディングを使用（`encoding='utf-8'`）
- インデントを設定して読みやすく（`indent=2`）
- 日本語を保持（`ensure_ascii=False`）

---

## 3.4 コード例2：材料分類による可視化

取得したデータを可視化することで、材料分類ごとの特徴を直感的に理解できます。

### 3.4.1 バンドギャップ分布の可視化

材料分類（金属、半導体、セラミックス）ごとのバンドギャップ分布を可視化します。

```python
# ===================================
# Example 2: 材料分類による物性データの可視化
# ===================================

import matplotlib.pyplot as plt
import numpy as np

def plot_band_gap_distribution(classified_materials: Dict[str, List[Dict]]):
    """
    材料分類ごとのバンドギャップ分布を可視化

    Args:
        classified_materials: 分類ごとの材料データ
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {
        "金属": "#FF6B6B",
        "半導体": "#4ECDC4",
        "セラミックス": "#45B7D1"
    }

    for mat_type in ["金属", "半導体", "セラミックス"]:
        if mat_type in classified_materials:
            materials = classified_materials[mat_type]
            band_gaps = [m["band_gap"] for m in materials]
            formulas = [m["formula"] for m in materials]

            # 散布図プロット
            ax.scatter(x_positions, band_gaps,
                      s=150, color=colors[mat_type],
                      alpha=0.7, label=mat_type)

    ax.set_ylabel('Band Gap (eV)')
    ax.set_xlabel('Material Type')
    ax.set_title('Band Gap Distribution by Material Classification')
    plt.savefig('output/chapter3/band_gap_distribution.png', dpi=150)
```

### 3.4.2 可視化結果の解釈

生成されたグラフから、以下のような材料分類の特徴が読み取れます：

**金属材料**:
- バンドギャップ = 0 eV（導電性が高い）
- 例: Al（アルミニウム）、Fe（鉄）

**半導体材料**:
- バンドギャップ = 0.7-3.5 eV（中程度）
- 例: Si（シリコン）1.14 eV、Ge（ゲルマニウム）0.74 eV

**セラミックス材料**:
- バンドギャップ = 5-7 eV（絶縁性が高い）
- 例: Al2O3（アルミナ）6.28 eV、SiO2（シリカ）5.61 eV

### 3.4.3 密度と生成エネルギーの関係

材料の安定性（生成エネルギー）と密度の関係を可視化します。

```python
def plot_density_vs_formation_energy(classified_materials):
    """
    密度と生成エネルギーの関係を材料分類ごとに可視化
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    for mat_type, materials in classified_materials.items():
        densities = [m["density"] for m in materials]
        formation_energies = [m["formation_energy"] for m in materials]

        ax.scatter(densities, formation_energies,
                  s=150, color=colors[mat_type],
                  alpha=0.7, label=mat_type)

    ax.set_xlabel('Density (g/cm³)')
    ax.set_ylabel('Formation Energy (eV/atom)')
    plt.savefig('output/chapter3/density_vs_formation_energy.png')
```

**実行結果**:
```
【セラミックス】
材料数: 2件

バンドギャップ (eV):
  最小: 5.610
  最大: 6.280
  平均: 5.945
  標準偏差: 0.335

密度 (g/cm³):
  最小: 2.65
  最大: 3.99
  平均: 3.32

生成エネルギー (eV/atom):
  最小: -16.574
  最大: -10.847
  平均: -13.710
```

**重要な洞察**:
- 生成エネルギーが低い（負の値が大きい）ほど、材料は安定
- セラミックス材料（Al2O3）は非常に安定（-16.574 eV/atom）
- 密度と安定性には必ずしも相関がない

---

## 3.5 実装のポイント

### 3.5.1 型ヒントの活用

Pythonの型ヒント（Type Hints）を使用することで、コードの可読性と保守性が向上します。

```python
from typing import Dict, List, Optional

def get_material_by_formula(formula: str) -> Optional[Dict]:
    """
    型ヒントにより、以下が明確になります：
    - 引数: formula は文字列型
    - 戻り値: 辞書型またはNone
    """
    pass
```

### 3.5.2 エラーハンドリング

実際のAPIを使用する場合、ネットワークエラーやデータの不整合に備える必要があります。

```python
def load_materials_data(filepath: str) -> List[Dict]:
    """JSONファイルから材料データを読み込み"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        print(f"Warning: {filepath} not found. Using sample data.")
        return get_sample_data()
```

### 3.5.3 データ検証

取得したデータの妥当性を確認することも重要です。

```python
def validate_material_data(material: Dict) -> bool:
    """
    材料データの妥当性を検証

    必須フィールド:
    - material_id
    - formula
    - band_gap (0以上)
    - density (正の値)
    """
    required_fields = ["material_id", "formula", "band_gap", "density"]

    # 必須フィールドの存在確認
    for field in required_fields:
        if field not in material:
            return False

    # 数値の妥当性確認
    if material["band_gap"] < 0:
        return False
    if material["density"] <= 0:
        return False

    return True
```

---

## 3.6 実践的な活用例

### 3.6.1 材料スクリーニング

特定の条件を満たす材料を検索する実用的な例です。

```python
def screen_materials(materials: List[Dict],
                    min_band_gap: float = 1.0,
                    max_band_gap: float = 3.0) -> List[Dict]:
    """
    バンドギャップの範囲で材料をスクリーニング

    半導体デバイス用途では、1.0-3.0 eVのバンドギャップが
    適していることが多い

    Args:
        materials: 材料データのリスト
        min_band_gap: 最小バンドギャップ (eV)
        max_band_gap: 最大バンドギャップ (eV)

    Returns:
        条件を満たす材料のリスト
    """
    screened = [
        m for m in materials
        if min_band_gap <= m["band_gap"] <= max_band_gap
    ]
    return screened
```

**使用例**:
```python
# 太陽電池材料の候補を検索（バンドギャップ 1.0-1.8 eV）
solar_cell_candidates = screen_materials(
    SAMPLE_MATERIALS_DATA,
    min_band_gap=1.0,
    max_band_gap=1.8
)

for material in solar_cell_candidates:
    print(f"{material['formula']}: {material['band_gap']} eV")
# 出力: Si: 1.14 eV
```

### 3.6.2 材料比較レポート

複数の材料を比較するレポート生成機能です。

```python
def generate_comparison_report(materials: List[Dict]) -> str:
    """
    材料比較レポートを生成

    Returns:
        マークダウン形式のレポート
    """
    report = "# 材料比較レポート\n\n"
    report += "| 化学式 | 分類 | バンドギャップ | 密度 | 安定性 |\n"
    report += "|--------|------|---------------|------|--------|\n"

    for m in materials:
        report += f"| {m['formula']} | "
        report += f"{m['material_type']} | "
        report += f"{m['band_gap']:.2f} eV | "
        report += f"{m['density']:.2f} g/cm³ | "
        report += f"{m['formation_energy']:.2f} eV/atom |\n"

    return report
```

---

## 3.7 まとめ

本章では、材料データベースを活用した実践的な実装方法を学びました。

### 本章で学んだこと

✅ **基本理解**:
- 材料データベース（Materials Project等）の概要と重要性
- APIを使用したデータ取得の基本手順
- 材料分類（金属、半導体、セラミックス）ごとの特徴

✅ **実践スキル**:
- Pythonで材料データを取得するコード実装
- JSONファイルを使ったデータの永続化
- matplotlib/numpyを使った物性データの可視化
- 材料分類ごとの統計分析

✅ **応用力**:
- 型ヒントとエラーハンドリングの活用
- 条件に基づく材料スクリーニング
- データ検証と品質管理

### 重要なポイント

**💡 データベースの活用価値**:
- 140,000件以上の材料データに即座にアクセス
- 実験を行わずに材料の物性を予測可能
- 新材料開発の時間とコストを大幅に削減

**🎯 実装のベストプラクティス**:
1. **型安全性**: 型ヒントで関数の入出力を明確化
2. **エラー処理**: 予期しない状況に備える
3. **データ検証**: 取得したデータの妥当性を確認
4. **可視化**: グラフで直感的な理解を促進

**⚠️ 注意点**:
- APIキーは秘密情報として扱う
- API利用制限（レート制限）に注意
- データの単位と精度を確認する
- 計算データと実験データの違いを理解する

### 次のステップ

本章で学んだデータベース活用技術を基に、次の章では実際の材料分類問題に取り組みます：

1. **第4章: 機械学習による材料分類**
   - 取得したデータを使った分類モデルの構築
   - 特徴量エンジニアリング
   - モデルの評価と最適化

2. **発展的な学習**:
   - 実際のMaterials Project APIの使用
   - より複雑なクエリとフィルタリング
   - 大規模データセットの処理

### 参考文献

#### 公式ドキュメント
1. Materials Project. (2024). "API Documentation." https://materialsproject.org/api
2. Materials Project. (2024). "Database Statistics." https://materialsproject.org/about
3. Python Software Foundation. (2024). "Type Hints (PEP 484)." https://peps.python.org/pep-0484/

#### 学術論文
1. Jain, A., et al. (2013). "Commentary: The Materials Project: A materials genome approach to accelerating materials innovation." *APL Materials*, 1(1), 011002.
2. Ong, S. P., et al. (2015). "The Materials Application Programming Interface (API): A simple, flexible and efficient API for materials data based on REpresentational State Transfer (REST) principles." *Computational Materials Science*, 97, 209-215.

#### チュートリアル・記事
1. Materials Project Workshop. (2023). "Getting Started with Materials Project API."
2. Python Data Science Handbook. (2023). "Visualization with Matplotlib."

---

**完全なコード例は以下のファイルで利用可能です**:
- `example1_materials_data_retrieval.py` - データ取得の完全実装
- `example2_materials_visualization.py` - 可視化の完全実装
- `verification_log.txt` - コード実行検証ログ

---

*本章の内容についてご質問やフィードバックがありましたら、お気軽にお問い合わせください。*
