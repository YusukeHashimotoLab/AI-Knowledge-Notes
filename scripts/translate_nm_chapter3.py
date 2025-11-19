#!/usr/bin/env python3
"""
Translate NM Chapter 3 from Japanese to English
Handles large file by processing in chunks
"""

import re
from pathlib import Path

# File paths
JP_FILE = "/Users/yusukehashimoto/Documents/pycharm/AI_Homepage/wp/knowledge/jp/MI/nm-introduction/chapter3-hands-on.html"
EN_FILE = "/Users/yusukehashimoto/Documents/pycharm/AI_Homepage/wp/knowledge/en/MI/nm-introduction/chapter3-hands-on.html"

# Translation dictionary for NM Chapter 3 content
TRANSLATIONS = {
    # Meta and header
    '<html lang="ja">': '<html lang="en">',
    '<title>Chapter 3: Python実践チュートリアル - AI Terakoya</title>': '<title>Chapter 3: Hands-On Python Tutorial - AI Terakoya</title>',

    # Breadcrumb
    'AI寺子屋トップ': 'AI Terakoya Top',
    'マテリアルズ・インフォマティクス': 'Materials Informatics',

    # Header content
    'Chapter 3: Python実践チュートリアル': 'Chapter 3: Hands-On Python Tutorial',
    'ナノ材料データ解析と機械学習': 'Nanomaterial Data Analysis and Machine Learning',
    '読了時間: 30-40分': 'Reading Time: 30-40 min',
    '難易度: 初級': 'Difficulty: Beginner',
    'コード例: 0個': 'Code Examples: 0',
    '演習問題: 0問': 'Exercises: 0',

    # Chapter description
    '小規模データでも効く回帰モデルとベイズ最適化で、効率よく条件探索する筋肉を付けます。MDデータの要点可視化とSHAPによる解釈まで一気に通します。': 'Build skills for efficiently exploring conditions using regression models effective even with small datasets and Bayesian optimization. Covers essential visualization of MD data and interpretation with SHAP in one go.',
    '💡 補足:': '💡 Supplement:',
    '少ない試行で良い条件を見つけるのが目標。ベイズ最適化は"金属探知機"的に当たりを導きます。': 'The goal is to find good conditions with minimal trials. Bayesian optimization guides you to hits like a "metal detector".',

    # Learning objectives
    '本章の学習目標': 'Learning Objectives',
    '本章を学習することで、以下のスキルを習得できます：': 'By studying this chapter, you will acquire the following skills:',
    'ナノ粒子データの生成・可視化・前処理の実践': 'Hands-on nanoparticle data generation, visualization, and preprocessing',
    '5種類の回帰モデルによるナノ材料物性予測': 'Prediction of nanomaterial properties using 5 types of regression models',
    'ベイズ最適化によるナノ材料の最適設計': 'Optimal design of nanomaterials through Bayesian optimization',
    'SHAP分析による機械学習モデルの解釈': 'Interpretation of machine learning models using SHAP analysis',
    '多目的最適化によるトレードオフ分析': 'Trade-off analysis through multi-objective optimization',
    'TEM画像解析とサイズ分布のフィッティング': 'TEM image analysis and size distribution fitting',
    '異常検知による品質管理への応用': 'Application to quality control through anomaly detection',

    # Section 3.1
    '3.1 環境構築': '3.1 Environment Setup',
    '必要なライブラリ': 'Required Libraries',
    '本チュートリアルで使用する主要なPythonライブラリ：': 'Main Python libraries used in this tutorial:',
    '# データ処理・可視化': '# Data processing and visualization',
    '# 機械学習': '# Machine learning',
    '# 最適化': '# Optimization',
    '# モデル解釈': '# Model interpretation',
    '# 多目的最適化（オプション）': '# Multi-objective optimization (optional)',

    'インストール方法': 'Installation Methods',
    'Option 1: Anaconda環境': 'Option 1: Anaconda Environment',
    '# Anacondaで新しい環境を作成': '# Create a new environment with Anaconda',
    '# 必要なライブラリをインストール': '# Install required libraries',
    '# 多目的最適化用（オプション）': '# For multi-objective optimization (optional)',

    'Option 2: venv + pip環境': 'Option 2: venv + pip Environment',
    '# 仮想環境を作成': '# Create virtual environment',
    '# 仮想環境を有効化': '# Activate virtual environment',

    'Option 3: Google Colab': 'Option 3: Google Colab',
    'Google Colabを使用する場合、以下のコードをセルで実行：': 'When using Google Colab, execute the following code in a cell:',
    '# 追加パッケージのインストール': '# Install additional packages',
    '# インポートの確認': '# Verify imports',
    '環境構築完了！': 'Environment setup complete!',

    # Section 3.2
    '3.2 ナノ粒子データの準備と可視化': '3.2 Nanoparticle Data Preparation and Visualization',
    '【例1】合成データ生成：金ナノ粒子のサイズと光学特性': '[Example 1] Synthetic Data Generation: Size and Optical Properties of Gold Nanoparticles',
    '金ナノ粒子の局在表面プラズモン共鳴（LSPR）波長は、粒子サイズに依存します。この関係を模擬データで表現します。': 'The localized surface plasmon resonance (LSPR) wavelength of gold nanoparticles depends on particle size. This relationship is represented with simulated data.',

    # Code comments
    '# 日本語フォント設定（必要に応じて）': '# Font settings (if needed)',
    '# 乱数シードの設定（再現性のため）': '# Set random seed (for reproducibility)',
    '# サンプル数': '# Number of samples',
    '# 金ナノ粒子のサイズ（nm）: 平均15 nm、標準偏差5 nm': '# Gold nanoparticle size (nm): mean 15 nm, std dev 5 nm',
    '# LSPR波長（nm）: Mie理論の簡易近似': '# LSPR wavelength (nm): simplified Mie theory approximation',
    '# 基本波長520 nm + サイズ依存項 + ノイズ': '# Base wavelength 520 nm + size-dependent term + noise',
    '# 合成条件': '# Synthesis conditions',
    '# 温度（℃）': '# Temperature (°C)',
    '# データフレームの作成': '# Create DataFrame',

    # Output messages
    '"金ナノ粒子データの生成完了"': '"Gold nanoparticle data generation complete"',
    '"\\n基本統計量:"': '"\\nBasic statistics:"',

    '【例2】サイズ分布のヒストグラム': '[Example 2] Size Distribution Histogram',
    '# サイズ分布のヒストグラム': '# Size distribution histogram',
    '# ヒストグラムとKDE（カーネル密度推定）': '# Histogram and KDE (kernel density estimation)',
    '# KDEプロット': '# KDE plot',
    '"平均サイズ: {data[\'size_nm\'].mean():.2f} nm"': '"Average size: {data[\'size_nm\'].mean():.2f} nm"',
    '"標準偏差: {data[\'size_nm\'].std():.2f} nm"': '"Standard deviation: {data[\'size_nm\'].std():.2f} nm"',
    '"中央値: {data[\'size_nm\'].median():.2f} nm"': '"Median: {data[\'size_nm\'].median():.2f} nm"',

    '【例3】散布図マトリックス': '[Example 3] Scatter Plot Matrix',
    '# ペアプロット（散布図マトリックス）': '# Pairplot (scatter plot matrix)',
    '"各変数間の関係を可視化しました"': '"Visualized relationships between variables"',

    '【例4】相関行列のヒートマップ': '[Example 4] Correlation Matrix Heatmap',
    '# 相関行列の計算': '# Calculate correlation matrix',
    '# ヒートマップ': '# Heatmap',
}

def translate_text(text: str) -> str:
    """Apply translations to text"""
    result = text
    for jp, en in TRANSLATIONS.items():
        result = result.replace(jp, en)
    return result

def main():
    print("Reading Japanese file...")
    with open(JP_FILE, 'r', encoding='utf-8') as f:
        content = f.read()

    print("Applying translations...")
    translated = translate_text(content)

    # Ensure output directory exists
    output_path = Path(EN_FILE)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Writing to {EN_FILE}...")
    with open(EN_FILE, 'w', encoding='utf-8') as f:
        f.write(translated)

    print("✓ Translation complete!")

    # Count remaining Japanese characters
    jp_pattern = re.compile(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF]+')
    jp_matches = jp_pattern.findall(translated)

    if jp_matches:
        print(f"\n⚠ Warning: {len(jp_matches)} Japanese text segments remain")
        print("First 10 occurrences:")
        for i, match in enumerate(jp_matches[:10], 1):
            print(f"  {i}. {match}")
    else:
        print("\n✓ No Japanese characters detected!")

    # Count lines
    lines = translated.split('\n')
    print(f"\nTotal lines: {len(lines)}")
    print(f"File size: {len(translated)} characters")

if __name__ == "__main__":
    main()
