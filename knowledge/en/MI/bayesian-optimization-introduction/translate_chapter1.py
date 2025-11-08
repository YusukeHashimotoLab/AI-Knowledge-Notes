#!/usr/bin/env python3
"""
Systematic translation script for chapter-1.html
Translates all remaining Japanese text to English using predefined mappings.
"""

import re
from pathlib import Path

# Translation mappings for common terms and phrases
translations = {
    # File/section headers
    "真の目的関数（未知と仮定）": "True objective function (assumed unknown)",
    "optimizationしたい特性（例：ionic conductivity）": "Property to optimize (e.g., ionic conductivity)",
    "ピークは x=0.7 付近にある": "Peak is around x=0.7",

    # Function definitions
    "random searchを実行": "Execute random search",
    "ランダムサンプリングによるexploration": "Exploration by random sampling",
    "サンプル数": "Number of samples",
    "exploration範囲": "Exploration range",
    "サンプリングした x 座標": "Sampled x coordinates",
    "観測値": "Observed values",

    # Simulation and execution
    "シミュレーション実行": "Run simulation",
    "20回の実験": "20 experiments",
    "真の最適値を計算（比較用）": "Calculate true optimal value (for comparison)",
    "可視化": "Visualization",

    # Plot labels
    "左図：explorationの様子": "Left plot: Exploration progress",
    "真の関数": "True function",
    "ランダムサンプル": "Random samples",
    "最良点": "Best point",
    "真の最適値": "True optimal value",
    "パラメータ x": "Parameter x",
    "特性値 y（例：ionic conductivity）": "Property value y (e.g., ionic conductivity)",
    "random searchの結果": "Random search results",
    "右図：最良値の推移": "Right plot: Best value progression",
    "これまでの最良値": "Best value so far",
    "実験回数": "Number of experiments",
    "explorationの進捗": "Exploration progress",

    # Results and summary
    "結果のサマリー": "Results summary",
    "回の実験": "experiments",
    "発見した最良値": "Best value found",
    "達成率": "Achievement rate",
    "最適値からの乖離": "Deviation from optimal value",
    "出力": "Output",

    # Problem analysis
    "random searchの問題点": "Problems with random search",
    "過去の実験結果をexploitationしない": "Does not exploit past experimental results",
    "良い結果が出た領域の周辺を集中的にexplorationしない": "Does not intensively explore regions that yielded good results",
    "悪い結果が出た領域も同じ確率でサンプリング": "Samples regions with poor results with equal probability",
    "explorationの偏り": "Biased exploration",
    "運が悪いと重要な領域をサンプリングしない": "May miss important regions with bad luck",
    "同じような場所を何度もサンプリングする可能性": "May sample similar locations multiple times",
    "収束が遅い": "Slow convergence",
    "実験回数を増やしても効率は改善しない": "Efficiency does not improve with more experiments",
    "の収束速度（n = サンプル数）": "convergence rate (n = number of samples)",

    # Grid search
    "grid searchの限界": "Limitations of grid search",
    "もう一つの古典的手法は": "Another classical approach is",
    "（格子exploration）です": "(grid exploration)",
    "grid searchと次元の呪い": "Grid search and the curse of dimensionality",
    "grid searchの計算コスト": "Computational cost of grid search",
    "grid searchの総サンプル数を計算": "Calculate total number of samples for grid search",
    "パラメータの次元数": "Number of parameter dimensions",
    "各次元あたりのグリッド点数": "Number of grid points per dimension",
    "総サンプル数": "Total number of samples",
    "次元数を変えて計算": "Calculate with varying dimensions",
    "各次元10点": "10 points per dimension",
    "1サンプル1時間": "1 hour per sample",
    "次元": "dimensions",
    "サンプル": "samples",
    "時間": "hours",
    "日": "days",
    "年": "years",
    "現実的な材料exploration問題": "Realistic materials exploration problems",
    "実際の材料exploration問題": "Actual materials exploration problems",
    "所要時間": "Time required",

    # Grid search problems
    "grid searchの問題点": "Problems with grid search",
    "次元の呪い": "Curse of dimensionality",
    "パラメータ数が増えると指数関数的にコスト増": "Cost increases exponentially with number of parameters",
    "計算資源の無駄": "Wasteful use of computational resources",
    "無意味な領域も均等にサンプリング": "Uniformly samples even meaningless regions",
    "柔軟性の欠如": "Lack of flexibility",
    "途中でexploration範囲を変更できない": "Cannot change exploration range mid-process",

    # Bayesian Optimization
    "Bayesian Optimizationの登場：賢いexploration戦略": "Introduction of Bayesian Optimization: Smart Exploration Strategy",
    "Bayesian Optimizationの基本アイデア": "Basic Idea of Bayesian Optimization",
    "は、上記の問題を解決する強力な手法です。": "is a powerful method that solves the above problems.",
    "核となる3つのアイデア": "Three core ideas",
    "代理モデル（Surrogate Model）": "Surrogate Model",
    "少数の観測データから目的関数の確率的モデルを構築": "Build a probabilistic model of the objective function from limited observations",
    "が一般的": "is commonly used",
    "次にどこをサンプリングすべきか決定": "Determines where to sample next",
    "とexploitation（exploitation）のバランス": "Balance between exploration and exploitation",
    "逐次的サンプリング": "Sequential sampling",
    "1回実験するたびにモデルを更新": "Update model after each experiment",
    "過去の結果を最大限exploitation": "Maximize exploitation of past results",

    # Workflow
    "Bayesian Optimizationのワークフロー": "Bayesian Optimization Workflow",
    "初期サンプリング": "Initial sampling",
    "少数のランダム実験": "Small number of random experiments",
    "代理モデル構築": "Build surrogate model",
    "Acquisition Functionのoptimization": "Optimization of Acquisition Function",
    "次の実験点を決定": "Determine next experimental point",
    "実験実行": "Execute experiment",
    "特性値を測定": "Measure property value",
    "終了条件?": "Termination condition?",
    "目標達成 or": "Goal achieved or",
    "予算上限": "Budget limit",
    "いいえ": "No",
    "はい": "Yes",
    "最良の材料を発見": "Discover best material",

    # Advantages
    "Bayesian Optimizationの利点": "Advantages of Bayesian Optimization",
    "少ない実験回数で最適解に到達": "Reach optimal solution with fewer experiments",
    "過去の実験結果をexploitation": "Exploit past experimental results",
    "（賢いexploration）": "(smart exploration)",
    "不確実性を考慮": "Consider uncertainty",
    "（explorationとexploitationのバランス）": "(balance of exploration and exploitation)",
    "並列化可能": "Parallelizable",
    "（複数の候補を同時提案）": "(propose multiple candidates simultaneously)",

    # Demo
    "Bayesian Optimizationの効率性のデモ": "Demonstration of Bayesian Optimization Efficiency",
    "Bayesian Optimization vs random searchの比較": "Comparison: Bayesian Optimization vs Random Search",
    "Bayesian Optimizationとrandom searchの効率比較": "Efficiency comparison: Bayesian Optimization vs Random Search",
    "注: 本格的な実装は第2章・第3章で扱います。ここでは概念的なデモ": "Note: Full implementation is covered in Chapters 2-3. This is a conceptual demo",
    "目的関数（未知と仮定）": "Objective function (assumed unknown)",
    "Li-ion batteryのionic conductivity（仮想的な例）": "Ionic conductivity of Li-ion battery (hypothetical example)",
    "簡易的なAcquisition Function（Upper Confidence Bound）": "Simplified Acquisition Function (Upper Confidence Bound)",
    "評価点": "Evaluation point",
    "学習済みGaussian Processモデル": "Trained Gaussian Process model",
    "explorationの強さ（大きいほどexploration重視）": "Exploration strength (larger values prioritize exploration)",
    "Bayesian Optimizationの簡易実装": "Simplified implementation of Bayesian Optimization",
    "Bayesian Optimizationのデモンストレーション": "Demonstration of Bayesian Optimization",
    "optimizationのイテレーション数": "Number of optimization iterations",
    "初期ランダムサンプル数": "Number of initial random samples",
    "サンプリングした点": "Sampled points",
    "初期ランダムサンプリング": "Initial random sampling",
    "Gaussian Processモデルの初期化": "Initialize Gaussian Process model",
    "逐次的サンプリング": "Sequential sampling",
    "Gaussian Processを学習": "Train Gaussian Process",
    "Acquisition Functionを最大化する点をexploration": "Explore point that maximizes Acquisition Function",
    "次の実験を実行": "Execute next experiment",
    "データに追加": "Add to data",
    "（比較用）": "(for comparison)",
    "最良値の推移を計算": "Calculate progression of best values",
    "explorationの様子": "Exploration progress",
    "exploration効率の比較": "Comparison of exploration efficiency",
    "改善率": "Improvement rate",
    "期待される出力": "Expected output",

    # Observations
    "重要な観察": "Key Observations",
    "Bayesian Optimizationは": "Bayesian Optimization",
    "少ない実験回数で真の最適値に近づく": "approaches the true optimal value with fewer experiments",
    "random searchは改善が頭打ちになる": "Random search plateaus in improvement",
    "有望な領域を集中的にexploration": "intensively explores promising regions",

    # Case studies
    "材料科学における成功事例": "Success Stories in Materials Science",
    "Li-ion batteryelectrolyteのoptimization": "Optimization of Li-ion Battery Electrolyte",
    "研究": "Research",
    "課題": "Challenge",
    "Li-ion batteryのelectrolyte配合をoptimization": "Optimize Li-ion battery electrolyte formulation",
    "ionic conductivityを最大化": "Maximize ionic conductivity",
    "（溶媒、塩、添加剤）": "(solvent, salt, additives)",
    "手法": "Method",
    "Bayesian Optimizationを適用": "Applied Bayesian Optimization",
    "random searchと比較": "Compared with random search",
    "結果": "Results",
    "6倍の効率向上": "6x efficiency improvement",
    "ionic conductivityが30%向上した配合を発見": "Discovered formulation with 30% improved ionic conductivity",
    "開発期間を数年から数ヶ月に短縮": "Reduced development time from years to months",
    "電池electrolyteoptimizationのシミュレーション": "Simulation of battery electrolyte optimization",
    "Li-ion batteryelectrolyteoptimizationのシミュレーション": "Simulation of Li-ion battery electrolyte optimization",
    "electrolyteのionic conductivityを計算（簡略化モデル）": "Calculate electrolyte ionic conductivity (simplified model)",
    "有機溶媒の混合比 (0-1)": "Organic solvent mixing ratio (0-1)",
    "Li塩の濃度 (0.5-2.0 M)": "Li salt concentration (0.5-2.0 M)",
    "添加剤の濃度 (0-5 wt%)": "Additive concentration (0-5 wt%)",
    "簡略化された経験式（実際はより複雑）": "Simplified empirical formula (actually more complex)",
    "溶媒効果（最適比は0.6付近）": "Solvent effect (optimal ratio around 0.6)",
    "塩濃度効果（最適は1.0 M付近）": "Salt concentration effect (optimal around 1.0 M)",
    "添加剤効果（少量で効果あり）": "Additive effect (effective in small amounts)",
    "ランダムノイズ（実験誤差）": "Random noise (experimental error)",
    "シミュレーション: random search": "Simulation: Random search",
    "ランダムに配合を選ぶ": "Randomly select formulations",
    "最良の配合を見つける": "Find best formulation",
    "random searchの結果 (100回の実験):": "Random search results (100 experiments):",
    "最高ionic conductivity:": "Maximum ionic conductivity:",
    "最適配合:": "Optimal formulation:",
    "溶媒混合比:": "Solvent mixing ratio:",
    "塩濃度:": "Salt concentration:",
    "添加剤濃度:": "Additive concentration:",
    "真の最適配合（参考）:": "True optimal formulation (reference):",

    # Common patterns
    "通り": "combinations",
    "刻み": "increments",
    "各次元": "each dimension",
    "点": "points",
}

def translate_japanese_text(text: str) -> str:
    """Translate Japanese text to English using the translation dictionary."""
    result = text
    for jp, en in translations.items():
        result = result.replace(jp, en)
    return result

def main():
    file_path = Path(__file__).parent / "chapter-1.html"

    print(f"Reading {file_path}...")
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    original_content = content

    # Apply translations
    print("Applying translations...")
    content = translate_japanese_text(content)

    # Count changes
    if content != original_content:
        print(f"Translations applied successfully!")
        print(f"Writing back to {file_path}...")
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print("Done!")
    else:
        print("No changes made.")

if __name__ == "__main__":
    main()
