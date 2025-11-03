"""
弾性変形のシミュレーション - Hooke's Law（フックの法則）

このスクリプトは、金属材料の弾性変形における
応力-ひずみ関係をシミュレートします。
"""

# ===================================
# Example 1: 弾性変形シミュレーション
# ===================================

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List

# 日本語フォント設定（グラフ用）
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False


class ElasticMaterial:
    """
    弾性材料のクラス

    Hooke's Lawに基づいて弾性変形をモデル化
    σ = E × ε

    Attributes:
        name (str): 材料名
        youngs_modulus (float): ヤング率 [GPa]
        poisson_ratio (float): ポアソン比 [-]
        yield_strength (float): 降伏強度 [MPa]
    """

    def __init__(self, name: str, youngs_modulus: float,
                 poisson_ratio: float, yield_strength: float):
        """
        弾性材料の初期化

        Args:
            name: 材料名
            youngs_modulus: ヤング率 [GPa]
            poisson_ratio: ポアソン比
            yield_strength: 降伏強度 [MPa]
        """
        self.name = name
        self.E = youngs_modulus * 1000  # GPaをMPaに変換
        self.nu = poisson_ratio
        self.sigma_y = yield_strength

    def calculate_stress(self, strain: np.ndarray) -> np.ndarray:
        """
        ひずみから応力を計算（Hooke's Law）

        σ = E × ε (弾性域内)

        Args:
            strain: ひずみ配列 [-]

        Returns:
            応力配列 [MPa]
        """
        # 弾性域のみ考慮（降伏強度まで）
        stress = self.E * strain

        # 降伏強度を超えないように制限
        stress = np.minimum(stress, self.sigma_y)

        return stress

    def calculate_strain_energy(self, strain: float) -> float:
        """
        ひずみエネルギー密度を計算

        U = (1/2) × σ × ε = (1/2) × E × ε²

        Args:
            strain: ひずみ [-]

        Returns:
            ひずみエネルギー密度 [MJ/m³]
        """
        if strain * self.E > self.sigma_y:
            # 降伏点でのひずみエネルギー
            epsilon_y = self.sigma_y / self.E
            energy = 0.5 * self.sigma_y * epsilon_y
        else:
            energy = 0.5 * self.E * strain**2

        return energy / 1e6  # MPaをMJ/m³に変換


def simulate_elastic_behavior(materials: List[ElasticMaterial],
                              max_strain: float = 0.005) -> None:
    """
    複数材料の弾性挙動をシミュレート

    Args:
        materials: 材料リスト
        max_strain: 最大ひずみ [-]
    """
    # ひずみ範囲を設定（0から最大ひずみまで）
    strain = np.linspace(0, max_strain, 1000)

    # プロット設定
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']

    for idx, material in enumerate(materials):
        # 応力計算
        stress = material.calculate_stress(strain)

        # 降伏点の特定
        yield_idx = np.where(stress >= material.sigma_y)[0]
        if len(yield_idx) > 0:
            yield_strain = strain[yield_idx[0]]
        else:
            yield_strain = max_strain

        # 応力-ひずみ曲線プロット
        ax1.plot(strain * 100, stress,
                linewidth=2.5, color=colors[idx % len(colors)],
                label=f'{material.name} (E={material.E/1000:.0f} GPa)')

        # 降伏点をマーク
        ax1.plot(yield_strain * 100, material.sigma_y,
                'o', markersize=10, color=colors[idx % len(colors)])

        # ひずみエネルギー計算
        energies = [material.calculate_strain_energy(s) for s in strain]
        ax2.plot(strain * 100, energies,
                linewidth=2.5, color=colors[idx % len(colors)],
                label=material.name)

    # 応力-ひずみ曲線の装飾
    ax1.set_xlabel('Strain (%)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Stress (MPa)', fontsize=12, fontweight='bold')
    ax1.set_title('Stress-Strain Curve (Elastic Region)',
                 fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(loc='lower right', fontsize=10)

    # ひずみエネルギー曲線の装飾
    ax2.set_xlabel('Strain (%)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Strain Energy Density (MJ/m³)',
                  fontsize=12, fontweight='bold')
    ax2.set_title('Strain Energy vs Strain',
                 fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.legend(loc='upper left', fontsize=10)

    plt.tight_layout()
    plt.savefig('output/topic2/chapter3/elastic_deformation.png', dpi=150)
    print("グラフを保存しました: output/topic2/chapter3/elastic_deformation.png")


def calculate_elastic_modulus_experiment(force: np.ndarray,
                                        displacement: np.ndarray,
                                        original_length: float,
                                        cross_section: float) -> Tuple[float, float]:
    """
    実験データからヤング率を計算

    Args:
        force: 荷重データ [N]
        displacement: 変位データ [mm]
        original_length: 元の長さ [mm]
        cross_section: 断面積 [mm²]

    Returns:
        (ヤング率 [GPa], R²値)
    """
    # 応力計算 [MPa]
    stress = force / cross_section

    # ひずみ計算 [-]
    strain = displacement / original_length

    # 線形回帰でヤング率を推定
    # σ = E × ε の傾きがヤング率
    coefficients = np.polyfit(strain, stress, 1)
    youngs_modulus = coefficients[0] / 1000  # GPaに変換

    # R²値を計算
    stress_fitted = np.polyval(coefficients, strain)
    ss_res = np.sum((stress - stress_fitted)**2)
    ss_tot = np.sum((stress - np.mean(stress))**2)
    r_squared = 1 - (ss_res / ss_tot)

    return youngs_modulus, r_squared


def main():
    """メイン実行関数"""

    print("=" * 70)
    print("弾性変形シミュレーション - Hooke's Law")
    print("=" * 70)

    # 代表的な金属材料のデータ
    materials = [
        ElasticMaterial("Steel (鋼)", 210, 0.30, 250),
        ElasticMaterial("Aluminum (アルミニウム)", 70, 0.33, 100),
        ElasticMaterial("Copper (銅)", 120, 0.34, 70),
        ElasticMaterial("Titanium (チタン)", 110, 0.34, 880)
    ]

    print("\n【材料物性データ】")
    print("-" * 70)
    for mat in materials:
        print(f"\n{mat.name}:")
        print(f"  ヤング率: {mat.E/1000:.0f} GPa")
        print(f"  ポアソン比: {mat.nu:.2f}")
        print(f"  降伏強度: {mat.sigma_y:.0f} MPa")

    # 弾性挙動のシミュレーション
    print("\n" + "=" * 70)
    print("弾性挙動シミュレーション実行中...")
    print("=" * 70)

    simulate_elastic_behavior(materials, max_strain=0.005)

    # 各材料の降伏点情報
    print("\n【降伏点情報】")
    print("-" * 70)
    for mat in materials:
        yield_strain = mat.sigma_y / mat.E
        yield_energy = mat.calculate_strain_energy(yield_strain)

        print(f"\n{mat.name}:")
        print(f"  降伏ひずみ: {yield_strain*100:.3f}%")
        print(f"  降伏点でのひずみエネルギー: {yield_energy:.3f} MJ/m³")

    # 実験データからのヤング率推定デモ
    print("\n" + "=" * 70)
    print("【実験データ解析例】")
    print("-" * 70)

    # 模擬実験データ（鋼の引張試験）
    # 試験片: 直径10mm, 標点間距離50mm
    diameter = 10.0  # mm
    gauge_length = 50.0  # mm
    cross_section = np.pi * (diameter/2)**2  # mm²

    # 荷重-変位データ（弾性域）
    displacement = np.linspace(0, 0.05, 20)  # mm
    true_E = 210  # GPa
    force = (true_E * 1000 * cross_section * displacement) / gauge_length  # N
    # ノイズ追加
    force += np.random.normal(0, 50, len(force))

    estimated_E, r2 = calculate_elastic_modulus_experiment(
        force, displacement, gauge_length, cross_section
    )

    print(f"\n試験片情報:")
    print(f"  直径: {diameter} mm")
    print(f"  標点間距離: {gauge_length} mm")
    print(f"  断面積: {cross_section:.2f} mm²")
    print(f"\n解析結果:")
    print(f"  推定ヤング率: {estimated_E:.1f} GPa")
    print(f"  R² 値: {r2:.4f}")
    print(f"  真の値: {true_E} GPa")
    print(f"  誤差: {abs(estimated_E - true_E)/true_E*100:.2f}%")

    print("\n" + "=" * 70)
    print("シミュレーション完了")
    print("=" * 70)


if __name__ == "__main__":
    main()
