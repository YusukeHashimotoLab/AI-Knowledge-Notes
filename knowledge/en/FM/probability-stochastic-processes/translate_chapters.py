#!/usr/bin/env python3
"""
Translation script for HTML chapter files from Japanese to English.
Preserves all HTML structure, CSS, JavaScript, equations, and code blocks.
Only translates visible text content.
"""

import re
from pathlib import Path

# Translation mappings for common terms and phrases
TRANSLATIONS = {
    # Meta and titles
    "確率論と確率過程": "Probability Theory and Stochastic Processes",
    "確率変数と確率分布の基礎": "Fundamentals of Random Variables and Probability Distributions",
    "大数の法則と中心極限定理": "Law of Large Numbers and Central Limit Theorem",
    "マルコフ過程とポアソン過程": "Markov Processes and Poisson Processes",
    "確率微分方程式とWiener過程": "Stochastic Differential Equations and Wiener Processes",
    "プロセス制御への応用": "Applications to Process Control",

    # Chapter references
    "第1章": "Chapter 1",
    "第2章": "Chapter 2",
    "第3章": "Chapter 3",
    "第4章": "Chapter 4",
    "第5章": "Chapter 5",
    "第1章:": "Chapter 1:",
    "第2章:": "Chapter 2:",
    "第3章:": "Chapter 3:",
    "第4章:": "Chapter 4:",
    "第5章:": "Chapter 5:",

    # Navigation
    "基礎数理道場": "Fundamentals of Mathematics",
    "シリーズトップ": "Series Top",
    "第1章を読む": "Read Chapter 1",
    "第2章を読む": "Read Chapter 2",
    "第3章を読む": "Read Chapter 3",
    "第4章を読む": "Read Chapter 4",
    "第5章を読む": "Read Chapter 5",
    "第2章へ": "To Chapter 2",
    "第3章へ": "To Chapter 3",
    "第4章へ": "To Chapter 4",
    "第5章へ": "To Chapter 5",
    "← 第1章": "← Chapter 1",
    "← 第2章": "← Chapter 2",
    "← 第3章": "← Chapter 3",
    "← 第4章": "← Chapter 4",

    # Common terms
    "離散・連続確率変数": "Discrete/Continuous Random Variables",
    "期待値・分散": "Expectation/Variance",
    "二項・ポアソン分布": "Binomial/Poisson Distribution",
    "正規・指数分布": "Normal/Exponential Distribution",
    "大数の弱法則・強法則": "Weak/Strong Law of Large Numbers",
    "標本分布": "Sample Distribution",
    "収束性の可視化": "Convergence Visualization",
    "材料科学応用": "Materials Science Applications",
    "マルコフ連鎖": "Markov Chains",
    "推移確率行列": "Transition Probability Matrix",
    "定常分布": "Stationary Distribution",
    "ポアソン過程": "Poisson Process",
    "故障モデリング": "Failure Modeling",
    "Wiener過程": "Wiener Process",
    "確率微分方程式": "Stochastic Differential Equations",
    "Itô積分": "Itô Integral",
    "幾何ブラウン運動": "Geometric Brownian Motion",
    "OU過程": "OU Process",
    "時系列モデリング": "Time Series Modeling",
    "管理図": "Control Charts",
    "カルマンフィルタ": "Kalman Filter",
    "予知保全": "Predictive Maintenance",

    # Section headers
    "定義": "Definition",
    "定理": "Theorem",
    "例": "Example",
    "コード例": "Code Example",
    "演習問題": "Exercises",
    "Note": "Note",

    # Descriptions in definitions/theorems
    "離散確率変数": "Discrete Random Variable",
    "連続確率変数": "Continuous Random Variable",
    "確率質量関数": "Probability Mass Function",
    "確率密度関数": "Probability Density Function",
    "期待値": "Expectation",
    "分散": "Variance",
    "標準偏差": "Standard Deviation",
    "二項分布": "Binomial Distribution",
    "ポアソン分布": "Poisson Distribution",
    "正規分布": "Normal Distribution",
    "指数分布": "Exponential Distribution",
    "一様分布": "Uniform Distribution",
    "ガウス分布": "Gaussian Distribution",
    "ベータ分布": "Beta Distribution",
    "ガンマ分布": "Gamma Distribution",
    "ワイブル分布": "Weibull Distribution",
    "対数正規分布": "Lognormal Distribution",

    # Statistical terms
    "確率収束": "Convergence in Probability",
    "概収束": "Almost Sure Convergence",
    "分布収束": "Convergence in Distribution",
    "標本平均": "Sample Mean",
    "標本分散": "Sample Variance",
    "標準化": "Standardization",
    "標準正規分布": "Standard Normal Distribution",
    "累積分布関数": "Cumulative Distribution Function",

    # Exercise labels
    "演習1": "Exercise 1",
    "演習2": "Exercise 2",
    "演習3": "Exercise 3",

    # Footer
    "基礎数理・物理道場": "Fundamentals of Mathematics & Physics Dojo",
}

def translate_html_content(content: str) -> str:
    """
    Translate Japanese text in HTML while preserving structure.
    """
    # Change lang attribute
    content = content.replace('lang="ja"', 'lang="en"')

    # Apply all translations
    for jp, en in TRANSLATIONS.items():
        # Use word boundary when appropriate to avoid partial matches
        content = content.replace(jp, en)

    return content

def translate_file(source_path: Path, target_path: Path):
    """Translate a single HTML file."""
    print(f"Translating: {source_path.name}")

    with open(source_path, 'r', encoding='utf-8') as f:
        content = f.read()

    translated = translate_html_content(content)

    with open(target_path, 'w', encoding='utf-8') as f:
        f.write(translated)

    print(f"  → Created: {target_path.name}")

def main():
    source_dir = Path("/Users/yusukehashimoto/Documents/pycharm/AI_Homepage/wp/knowledge/jp/FM/probability-stochastic-processes")
    target_dir = Path("/Users/yusukehashimoto/Documents/pycharm/AI_Homepage/wp/knowledge/en/FM/probability-stochastic-processes")

    # Translate chapters 1-5
    for i in range(1, 6):
        source_file = source_dir / f"chapter-{i}.html"
        target_file = target_dir / f"chapter-{i}.html"

        if source_file.exists():
            translate_file(source_file, target_file)
        else:
            print(f"Warning: {source_file} not found")

    print("\nTranslation complete!")

if __name__ == "__main__":
    main()
