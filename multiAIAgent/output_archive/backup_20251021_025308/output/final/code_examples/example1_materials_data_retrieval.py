"""
材料データベースからのデータ取得例

このスクリプトは、Materials Project APIを使用して
材料物性データを取得する基本的な方法を示します。

注意: 実際に実行するにはMaterials Project APIキーが必要です
https://materialsproject.org/ でアカウント登録してAPIキーを取得してください
"""

# ===================================
# Example 1: 材料データベースAPIの基本的な使用方法
# ===================================

# 必要なライブラリのインポート
import json
from typing import Dict, List, Optional

# シミュレーション用のサンプルデータ
# 実際のMaterials Project APIから取得されるデータ構造を模擬
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
    {
        "material_id": "mp-149",
        "formula": "Al",
        "material_type": "金属",
        "band_gap": 0.0,
        "formation_energy": -3.747,
        "density": 2.70,
        "structure_type": "fcc",
        "space_group": "Fm-3m"
    },
    {
        "material_id": "mp-1143",
        "formula": "Fe",
        "material_type": "金属",
        "band_gap": 0.0,
        "formation_energy": -8.127,
        "density": 7.87,
        "structure_type": "bcc",
        "space_group": "Im-3m"
    },
    {
        "material_id": "mp-804",
        "formula": "Al2O3",
        "material_type": "セラミックス",
        "band_gap": 6.28,
        "formation_energy": -16.574,
        "density": 3.99,
        "structure_type": "corundum",
        "space_group": "R-3c"
    },
    {
        "material_id": "mp-1265",
        "formula": "SiO2",
        "material_type": "セラミックス",
        "band_gap": 5.61,
        "formation_energy": -10.847,
        "density": 2.65,
        "structure_type": "quartz",
        "space_group": "P3221"
    }
]


def get_material_by_formula(formula: str) -> Optional[Dict]:
    """
    化学式を指定して材料データを取得

    実際のAPIでは、Materials Projectのmpy.queryメソッドを使用します：
    mpr = MPRester("YOUR_API_KEY")
    data = mpr.query(criteria={"pretty_formula": formula}, ...)

    Args:
        formula (str): 化学式（例: "Si", "Al2O3"）

    Returns:
        Optional[Dict]: 材料データの辞書、見つからない場合はNone
    """
    # サンプルデータから該当する材料を検索
    for material in SAMPLE_MATERIALS_DATA:
        if material["formula"] == formula:
            return material

    return None


def get_materials_by_type(material_type: str) -> List[Dict]:
    """
    材料分類で材料データを検索

    Args:
        material_type (str): 材料分類（"金属", "セラミックス", "半導体"）

    Returns:
        List[Dict]: 該当する材料データのリスト
    """
    # 指定された材料分類に一致するデータをフィルタリング
    filtered_materials = [
        material for material in SAMPLE_MATERIALS_DATA
        if material["material_type"] == material_type
    ]

    return filtered_materials


def display_material_info(material: Dict) -> None:
    """
    材料情報を見やすく表示

    Args:
        material (Dict): 材料データの辞書
    """
    print(f"\n{'='*50}")
    print(f"Material ID: {material['material_id']}")
    print(f"化学式: {material['formula']}")
    print(f"材料分類: {material['material_type']}")
    print(f"{'='*50}")
    print(f"バンドギャップ: {material['band_gap']:.2f} eV")
    print(f"生成エネルギー: {material['formation_energy']:.3f} eV/atom")
    print(f"密度: {material['density']:.2f} g/cm³")
    print(f"結晶構造: {material['structure_type']}")
    print(f"空間群: {material['space_group']}")
    print(f"{'='*50}\n")


def main():
    """メイン実行関数"""

    print("=" * 60)
    print("材料データベースからのデータ取得デモ")
    print("=" * 60)

    # ===================================
    # 1. 化学式で材料を検索
    # ===================================
    print("\n【1】化学式による検索")
    print("-" * 60)

    # シリコンを検索
    si_data = get_material_by_formula("Si")
    if si_data:
        display_material_info(si_data)

    # アルミナを検索
    al2o3_data = get_material_by_formula("Al2O3")
    if al2o3_data:
        display_material_info(al2o3_data)

    # ===================================
    # 2. 材料分類で検索
    # ===================================
    print("\n【2】材料分類による検索")
    print("-" * 60)

    # 金属材料を全て取得
    metals = get_materials_by_type("金属")
    print(f"\n金属材料: {len(metals)}件")
    for metal in metals:
        print(f"  - {metal['formula']}: {metal['structure_type']} structure")

    # セラミックス材料を全て取得
    ceramics = get_materials_by_type("セラミックス")
    print(f"\nセラミックス材料: {len(ceramics)}件")
    for ceramic in ceramics:
        print(f"  - {ceramic['formula']}: バンドギャップ {ceramic['band_gap']:.2f} eV")

    # ===================================
    # 3. データの統計情報
    # ===================================
    print("\n【3】データセット統計")
    print("-" * 60)

    # 材料分類ごとの件数
    material_types = {}
    for material in SAMPLE_MATERIALS_DATA:
        mat_type = material["material_type"]
        material_types[mat_type] = material_types.get(mat_type, 0) + 1

    print("\n材料分類の分布:")
    for mat_type, count in material_types.items():
        print(f"  {mat_type}: {count}件")

    # バンドギャップの範囲
    band_gaps = [m["band_gap"] for m in SAMPLE_MATERIALS_DATA]
    print(f"\nバンドギャップの範囲:")
    print(f"  最小: {min(band_gaps):.2f} eV")
    print(f"  最大: {max(band_gaps):.2f} eV")
    print(f"  平均: {sum(band_gaps)/len(band_gaps):.2f} eV")

    # ===================================
    # 4. JSONファイルへの保存
    # ===================================
    print("\n【4】データの保存")
    print("-" * 60)

    output_file = "output/chapter3/materials_data.json"

    # データをJSON形式で保存
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(SAMPLE_MATERIALS_DATA, f, ensure_ascii=False, indent=2)

    print(f"\n材料データを保存しました: {output_file}")
    print(f"保存された材料数: {len(SAMPLE_MATERIALS_DATA)}件")


if __name__ == "__main__":
    main()
