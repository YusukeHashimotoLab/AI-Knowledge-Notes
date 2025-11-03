"""
塑性変形のモデリング - 降伏強度と加工硬化

このスクリプトは、金属材料の塑性変形における
降伏現象と加工硬化（Work Hardening）をモデル化します。
"""

# ===================================
# Example 2: 塑性変形モデリング
# ===================================

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Dict

# 日本語フォント設定
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False


class PlasticMaterial:
    """
    塑性変形を考慮した材料モデル

    Ludwik-Hollomon式による加工硬化モデル:
    σ = σ_y + K × ε_p^n

    Attributes:
        name (str): 材料名
        E (float): ヤング率 [MPa]
        sigma_y (float): 降伏強度 [MPa]
        K (float): 強度係数 [MPa]
        n (float): 加工硬化指数 [-]
    """

    def __init__(self, name: str, youngs_modulus: float,
                 yield_strength: float, strength_coef: float,
                 hardening_exp: float):
        """
        塑性材料の初期化

        Args:
            name: 材料名
            youngs_modulus: ヤング率 [GPa]
            yield_strength: 降伏強度 [MPa]
            strength_coef: 強度係数 K [MPa]
            hardening_exp: 加工硬化指数 n [-]
        """
        self.name = name
        self.E = youngs_modulus * 1000  # GPa → MPa
        self.sigma_y = yield_strength
        self.K = strength_coef
        self.n = hardening_exp

    def calculate_stress_strain(self, max_strain: float,
                               num_points: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """
        応力-ひずみ曲線を計算（弾性域+塑性域）

        Args:
            max_strain: 最大ひずみ [-]
            num_points: 計算点数

        Returns:
            (ひずみ配列, 応力配列)
        """
        strain = np.linspace(0, max_strain, num_points)
        stress = np.zeros_like(strain)

        # 降伏ひずみ
        epsilon_y = self.sigma_y / self.E

        for i, eps in enumerate(strain):
            if eps <= epsilon_y:
                # 弾性域: σ = E × ε
                stress[i] = self.E * eps
            else:
                # 塑性域: σ = σ_y + K × ε_p^n
                eps_plastic = eps - epsilon_y
                stress[i] = self.sigma_y + self.K * eps_plastic**self.n

        return strain, stress

    def calculate_true_stress_strain(self, engineering_strain: np.ndarray,
                                    engineering_stress: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        公称応力-ひずみから真応力-ひずみに変換

        真ひずみ: ε_true = ln(1 + ε_eng)
        真応力: σ_true = σ_eng × (1 + ε_eng)

        Args:
            engineering_strain: 公称ひずみ
            engineering_stress: 公称応力

        Returns:
            (真ひずみ, 真応力)
        """
        true_strain = np.log(1 + engineering_strain)
        true_stress = engineering_stress * (1 + engineering_strain)

        return true_strain, true_stress


def simulate_work_hardening(materials: list) -> None:
    """
    加工硬化挙動のシミュレーション

    Args:
        materials: 材料リスト
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    max_strain = 0.20  # 20%まで

    for idx, mat in enumerate(materials):
        # 応力-ひずみ曲線計算
        strain, stress = mat.calculate_stress_strain(max_strain)

        # 公称応力-ひずみ曲線
        ax1.plot(strain * 100, stress,
                linewidth=2.5, color=colors[idx],
                label=f'{mat.name} (n={mat.n:.2f})')

        # 降伏点マーク
        epsilon_y = mat.sigma_y / mat.E
        ax1.plot(epsilon_y * 100, mat.sigma_y,
                'o', markersize=10, color=colors[idx])

        # 真応力-ひずみ曲線
        true_strain, true_stress = mat.calculate_true_stress_strain(strain, stress)
        ax2.plot(true_strain * 100, true_stress,
                linewidth=2.5, color=colors[idx],
                label=mat.name)

        # 加工硬化率の計算 (dσ/dε)
        hardening_rate = np.gradient(stress, strain)
        ax3.plot(strain * 100, hardening_rate,
                linewidth=2.5, color=colors[idx],
                label=mat.name)

        # 塑性ひずみエネルギー
        plastic_work = np.cumsum(stress[1:] * np.diff(strain))
        ax4.plot(strain[1:] * 100, plastic_work / 1e6,
                linewidth=2.5, color=colors[idx],
                label=mat.name)

    # グラフ装飾
    ax1.set_xlabel('Engineering Strain (%)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Engineering Stress (MPa)', fontsize=11, fontweight='bold')
    ax1.set_title('Engineering Stress-Strain Curve',
                 fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=9)

    ax2.set_xlabel('True Strain (%)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('True Stress (MPa)', fontsize=11, fontweight='bold')
    ax2.set_title('True Stress-Strain Curve',
                 fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=9)

    ax3.set_xlabel('Strain (%)', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Work Hardening Rate (MPa)', fontsize=11, fontweight='bold')
    ax3.set_title('Work Hardening Rate vs Strain',
                 fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=9)

    ax4.set_xlabel('Strain (%)', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Plastic Work (MJ/m³)', fontsize=11, fontweight='bold')
    ax4.set_title('Plastic Work vs Strain',
                 fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig('output/topic2/chapter3/plastic_deformation.png', dpi=150)
    print("グラフを保存しました: output/topic2/chapter3/plastic_deformation.png")


def analyze_necking_instability(material: PlasticMaterial) -> Dict:
    """
    ネッキング（くびれ）発生条件を解析

    Considère条件: dσ/dε = σ のとき不安定化

    Args:
        material: 塑性材料

    Returns:
        ネッキング解析結果
    """
    # 広範囲のひずみで計算
    strain, stress = material.calculate_stress_strain(0.5, 5000)

    # 加工硬化率計算
    hardening_rate = np.gradient(stress, strain)

    # ネッキング条件: dσ/dε ≈ σ
    # 差が最小となる点を探す
    instability_criterion = np.abs(hardening_rate - stress)
    necking_idx = np.argmin(instability_criterion[100:]) + 100  # 初期値を除外

    necking_strain = strain[necking_idx]
    necking_stress = stress[necking_idx]

    # 真応力-ひずみに変換
    true_strain, true_stress = material.calculate_true_stress_strain(
        strain, stress
    )

    return {
        'necking_strain': necking_strain,
        'necking_stress': necking_stress,
        'true_necking_strain': true_strain[necking_idx],
        'true_necking_stress': true_stress[necking_idx],
        'ultimate_tensile_strength': necking_stress
    }


def main():
    """メイン実行関数"""

    print("=" * 70)
    print("塑性変形モデリング - 降伏と加工硬化")
    print("=" * 70)

    # 異なる加工硬化特性を持つ材料
    materials = [
        PlasticMaterial("Low Carbon Steel (軟鋼)",
                       210, 250, 600, 0.20),
        PlasticMaterial("Stainless Steel (ステンレス鋼)",
                       200, 300, 1000, 0.35),
        PlasticMaterial("Aluminum Alloy (アルミ合金)",
                       70, 100, 350, 0.15)
    ]

    print("\n【材料の加工硬化特性】")
    print("-" * 70)
    for mat in materials:
        print(f"\n{mat.name}:")
        print(f"  ヤング率 E: {mat.E/1000:.0f} GPa")
        print(f"  降伏強度 σ_y: {mat.sigma_y:.0f} MPa")
        print(f"  強度係数 K: {mat.K:.0f} MPa")
        print(f"  加工硬化指数 n: {mat.n:.2f}")

    # 加工硬化シミュレーション
    print("\n" + "=" * 70)
    print("加工硬化シミュレーション実行中...")
    print("=" * 70)

    simulate_work_hardening(materials)

    # ネッキング解析
    print("\n【ネッキング（不安定化）解析】")
    print("-" * 70)

    for mat in materials:
        result = analyze_necking_instability(mat)

        print(f"\n{mat.name}:")
        print(f"  ネッキング発生ひずみ: {result['necking_strain']*100:.2f}%")
        print(f"  引張強さ (UTS): {result['ultimate_tensile_strength']:.1f} MPa")
        print(f"  真ひずみ: {result['true_necking_strain']:.3f}")
        print(f"  真応力: {result['true_necking_stress']:.1f} MPa")

    # 加工硬化指数nの影響
    print("\n【加工硬化指数 n の影響】")
    print("-" * 70)
    print("\nn が大きいほど:")
    print("  ✓ 加工硬化が顕著（変形しにくくなる）")
    print("  ✓ ネッキング発生が遅れる（延性が向上）")
    print("  ✓ 引張強さが向上")
    print("\nn が小さいほど:")
    print("  ✓ 加工硬化が緩やか")
    print("  ✓ 早期にネッキング発生")
    print("  ✓ 延性が低下")

    # 実用的な加工硬化指数の例
    print("\n【代表的な材料の n 値】")
    print("-" * 70)
    typical_n_values = {
        "軟鋼": "0.20 - 0.25",
        "ステンレス鋼": "0.30 - 0.50",
        "アルミニウム合金": "0.10 - 0.20",
        "銅": "0.30 - 0.40",
        "チタン": "0.05 - 0.10"
    }

    for material, n_range in typical_n_values.items():
        print(f"  {material}: {n_range}")

    print("\n" + "=" * 70)
    print("シミュレーション完了")
    print("=" * 70)


if __name__ == "__main__":
    main()
