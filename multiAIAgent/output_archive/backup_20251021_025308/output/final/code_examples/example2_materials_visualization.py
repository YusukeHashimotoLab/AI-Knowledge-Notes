"""
材料分類による物性データの可視化

このスクリプトは、材料分類（金属、セラミックス、半導体）ごとに
物性データを可視化する方法を示します。
"""

# ===================================
# Example 2: 材料分類による物性データの可視化
# ===================================

# 必要なライブラリのインポート
import json
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List

# 日本語フォント設定（グラフの日本語表示用）
plt.rcParams['font.family'] = 'DejaVu Sans'  # システムにあるフォントを使用
plt.rcParams['axes.unicode_minus'] = False  # マイナス記号の文字化け防止


def load_materials_data(filepath: str) -> List[Dict]:
    """
    JSONファイルから材料データを読み込み

    Args:
        filepath (str): JSONファイルのパス

    Returns:
        List[Dict]: 材料データのリスト
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        # ファイルが存在しない場合はサンプルデータを返す
        print(f"Warning: {filepath} not found. Using sample data.")
        return get_sample_data()


def get_sample_data() -> List[Dict]:
    """サンプルデータを生成（ファイル読み込み失敗時のフォールバック）"""
    return [
        {
            "material_id": "mp-66",
            "formula": "Si",
            "material_type": "半導体",
            "band_gap": 1.14,
            "formation_energy": -5.424,
            "density": 2.33
        },
        {
            "material_id": "mp-149",
            "formula": "Al",
            "material_type": "金属",
            "band_gap": 0.0,
            "formation_energy": -3.747,
            "density": 2.70
        },
        {
            "material_id": "mp-1143",
            "formula": "Fe",
            "material_type": "金属",
            "band_gap": 0.0,
            "formation_energy": -8.127,
            "density": 7.87
        },
        {
            "material_id": "mp-804",
            "formula": "Al2O3",
            "material_type": "セラミックス",
            "band_gap": 6.28,
            "formation_energy": -16.574,
            "density": 3.99
        },
        {
            "material_id": "mp-1265",
            "formula": "SiO2",
            "material_type": "セラミックス",
            "band_gap": 5.61,
            "formation_energy": -10.847,
            "density": 2.65
        },
        {
            "material_id": "mp-32",
            "formula": "Ge",
            "material_type": "半導体",
            "band_gap": 0.744,
            "formation_energy": -4.632,
            "density": 5.32
        },
        {
            "material_id": "mp-2534",
            "formula": "GaN",
            "material_type": "半導体",
            "band_gap": 3.20,
            "formation_energy": -3.895,
            "density": 6.15
        },
        {
            "material_id": "mp-72",
            "formula": "Cu",
            "material_type": "金属",
            "band_gap": 0.0,
            "formation_energy": -3.721,
            "density": 8.96
        }
    ]


def classify_materials(materials: List[Dict]) -> Dict[str, List[Dict]]:
    """
    材料を分類ごとにグループ化

    Args:
        materials (List[Dict]): 材料データのリスト

    Returns:
        Dict[str, List[Dict]]: 分類ごとにグループ化された辞書
    """
    classified = {}

    for material in materials:
        mat_type = material["material_type"]
        if mat_type not in classified:
            classified[mat_type] = []
        classified[mat_type].append(material)

    return classified


def plot_band_gap_distribution(classified_materials: Dict[str, List[Dict]]) -> None:
    """
    材料分類ごとのバンドギャップ分布を可視化

    Args:
        classified_materials (Dict[str, List[Dict]]): 分類ごとの材料データ
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # 各分類のバンドギャップデータを抽出
    positions = []
    labels = []
    colors = {
        "金属": "#FF6B6B",
        "半導体": "#4ECDC4",
        "セラミックス": "#45B7D1"
    }

    pos = 1
    for mat_type in ["金属", "半導体", "セラミックス"]:
        if mat_type in classified_materials:
            materials = classified_materials[mat_type]
            band_gaps = [m["band_gap"] for m in materials]
            formulas = [m["formula"] for m in materials]

            # 各材料のバンドギャップをプロット
            x_positions = [pos + i*0.2 for i in range(len(band_gaps))]
            ax.scatter(x_positions, band_gaps, s=150,
                      color=colors[mat_type], alpha=0.7,
                      label=mat_type)

            # 化学式をアノテーション
            for x, y, formula in zip(x_positions, band_gaps, formulas):
                ax.annotate(formula, (x, y),
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=9)

            positions.append(pos + 0.2)
            labels.append(mat_type)
            pos += 1.5

    ax.set_ylabel('Band Gap (eV)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Material Type', fontsize=12, fontweight='bold')
    ax.set_title('Band Gap Distribution by Material Classification',
                fontsize=14, fontweight='bold')
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='upper left', fontsize=10)

    plt.tight_layout()
    plt.savefig('output/chapter3/band_gap_distribution.png', dpi=150)
    print("グラフを保存しました: output/chapter3/band_gap_distribution.png")


def plot_density_vs_formation_energy(classified_materials: Dict[str, List[Dict]]) -> None:
    """
    密度と生成エネルギーの関係を材料分類ごとに可視化

    Args:
        classified_materials (Dict[str, List[Dict]]): 分類ごとの材料データ
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {
        "金属": "#FF6B6B",
        "半導体": "#4ECDC4",
        "セラミックス": "#45B7D1"
    }

    markers = {
        "金属": "o",
        "半導体": "s",
        "セラミックス": "^"
    }

    # 各分類のデータをプロット
    for mat_type, materials in classified_materials.items():
        densities = [m["density"] for m in materials]
        formation_energies = [m["formation_energy"] for m in materials]
        formulas = [m["formula"] for m in materials]

        ax.scatter(densities, formation_energies,
                  s=150, color=colors[mat_type],
                  marker=markers[mat_type],
                  alpha=0.7, label=mat_type)

        # 化学式をアノテーション
        for x, y, formula in zip(densities, formation_energies, formulas):
            ax.annotate(formula, (x, y),
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=9)

    ax.set_xlabel('Density (g/cm³)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Formation Energy (eV/atom)', fontsize=12, fontweight='bold')
    ax.set_title('Density vs Formation Energy by Material Classification',
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='lower right', fontsize=10)

    plt.tight_layout()
    plt.savefig('output/chapter3/density_vs_formation_energy.png', dpi=150)
    print("グラフを保存しました: output/chapter3/density_vs_formation_energy.png")


def plot_material_type_summary(classified_materials: Dict[str, List[Dict]]) -> None:
    """
    材料分類ごとの統計サマリーを棒グラフで表示

    Args:
        classified_materials (Dict[str, List[Dict]]): 分類ごとの材料データ
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    material_types = list(classified_materials.keys())
    colors = ["#FF6B6B", "#4ECDC4", "#45B7D1"]

    # 左側: 材料数
    counts = [len(classified_materials[mt]) for mt in material_types]
    ax1.bar(material_types, counts, color=colors, alpha=0.7, edgecolor='black')
    ax1.set_ylabel('Number of Materials', fontsize=12, fontweight='bold')
    ax1.set_title('Material Count by Classification', fontsize=13, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3, linestyle='--')

    # 各バーの上に数値を表示
    for i, (mt, count) in enumerate(zip(material_types, counts)):
        ax1.text(i, count + 0.1, str(count),
                ha='center', va='bottom', fontweight='bold')

    # 右側: 平均バンドギャップ
    avg_band_gaps = []
    for mt in material_types:
        materials = classified_materials[mt]
        avg_bg = np.mean([m["band_gap"] for m in materials])
        avg_band_gaps.append(avg_bg)

    ax2.bar(material_types, avg_band_gaps, color=colors, alpha=0.7, edgecolor='black')
    ax2.set_ylabel('Average Band Gap (eV)', fontsize=12, fontweight='bold')
    ax2.set_title('Average Band Gap by Classification', fontsize=13, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3, linestyle='--')

    # 各バーの上に数値を表示
    for i, (mt, avg_bg) in enumerate(zip(material_types, avg_band_gaps)):
        ax2.text(i, avg_bg + 0.1, f'{avg_bg:.2f}',
                ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig('output/chapter3/material_type_summary.png', dpi=150)
    print("グラフを保存しました: output/chapter3/material_type_summary.png")


def print_statistics(classified_materials: Dict[str, List[Dict]]) -> None:
    """
    材料分類ごとの統計情報を表示

    Args:
        classified_materials (Dict[str, List[Dict]]): 分類ごとの材料データ
    """
    print("\n" + "="*70)
    print("材料分類ごとの統計情報")
    print("="*70)

    for mat_type, materials in classified_materials.items():
        print(f"\n【{mat_type}】")
        print("-" * 70)
        print(f"材料数: {len(materials)}件")

        # バンドギャップ統計
        band_gaps = [m["band_gap"] for m in materials]
        print(f"\nバンドギャップ (eV):")
        print(f"  最小: {min(band_gaps):.3f}")
        print(f"  最大: {max(band_gaps):.3f}")
        print(f"  平均: {np.mean(band_gaps):.3f}")
        print(f"  標準偏差: {np.std(band_gaps):.3f}")

        # 密度統計
        densities = [m["density"] for m in materials]
        print(f"\n密度 (g/cm³):")
        print(f"  最小: {min(densities):.2f}")
        print(f"  最大: {max(densities):.2f}")
        print(f"  平均: {np.mean(densities):.2f}")

        # 生成エネルギー統計
        formation_energies = [m["formation_energy"] for m in materials]
        print(f"\n生成エネルギー (eV/atom):")
        print(f"  最小: {min(formation_energies):.3f}")
        print(f"  最大: {max(formation_energies):.3f}")
        print(f"  平均: {np.mean(formation_energies):.3f}")


def main():
    """メイン実行関数"""

    print("=" * 70)
    print("材料分類による物性データ可視化デモ")
    print("=" * 70)

    # データの読み込み
    materials = load_materials_data("output/chapter3/materials_data.json")
    print(f"\n読み込んだ材料数: {len(materials)}件")

    # 材料を分類ごとにグループ化
    classified_materials = classify_materials(materials)

    # 統計情報の表示
    print_statistics(classified_materials)

    # グラフの生成
    print("\n" + "="*70)
    print("可視化グラフを生成中...")
    print("="*70 + "\n")

    plot_band_gap_distribution(classified_materials)
    plot_density_vs_formation_energy(classified_materials)
    plot_material_type_summary(classified_materials)

    print("\n" + "="*70)
    print("可視化完了！")
    print("="*70)


if __name__ == "__main__":
    main()
