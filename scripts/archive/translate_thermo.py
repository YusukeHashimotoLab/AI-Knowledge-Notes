#!/usr/bin/env python3
"""
Translate equilibrium-thermodynamics HTML files from Japanese to English
Preserves all structure, equations, and code blocks
"""

import re
from pathlib import Path

# Translation dictionary for common terms
translations = {
    # Meta and headers
    '基礎数理道場': 'Fundamentals Dojo',
    'AI寺子屋トップ': 'AI Terakoya Top',
    '基礎数理': 'Fundamentals',

    # Chapter titles
    '第1章': 'Chapter 1',
    '第2章': 'Chapter 2',
    '第3章': 'Chapter 3',
    '第4章': 'Chapter 4',
    '第5章': 'Chapter 5',

    # Chapter 1
    '熱力学の基本法則と熱力学ポテンシャル': 'Fundamental Laws of Thermodynamics and Thermodynamic Potentials',
    '熱力学の法則と熱力学関数': 'Laws of Thermodynamics and Thermodynamic Functions',

    # Chapter 2
    'Maxwell関係式と熱力学的関係': 'Maxwell Relations and Thermodynamic Relations',
    '相平衡とGibbs相律': 'Phase Equilibrium and Gibbs Phase Rule',

    # Chapter 3
    '相平衡と相図': 'Phase Equilibrium and Phase Diagrams',
    '化学ポテンシャルと相図': 'Chemical Potential and Phase Diagrams',

    # Chapter 4
    '相転移の分類とLandau理論': 'Classification of Phase Transitions and Landau Theory',
    '臨界現象とスケーリング理論': 'Critical Phenomena and Scaling Theory',

    # Chapter 5
    '材料科学への応用': 'Applications to Materials Science',

    # Common UI elements
    'シリーズTOP': 'Series Top',
    '学習目標': 'Learning Objectives',
    'シリーズ概要': 'Series Overview',
    '前提知識': 'Prerequisites',
    '演習問題': 'Exercises',
    'まとめ': 'Summary',
    'コード例': 'Code Example',

    # Numbers and metadata
    '章': ' Chapters',
    'コード例': ' Code Examples',
    '分': ' min',
    '中級': 'Intermediate',

    # Breadcrumb/nav
    'を読む': '',  # Will handle separately
    '第': 'Chapter ',

    # Common phrases
    '熱力学の第0〜第3法則の意味と応用を理解する': 'Understand the meaning and applications of the zeroth through third laws of thermodynamics',
    '内部エネルギー (U) の定義と熱力学第一法則を学ぶ': 'Learn the definition of internal energy (U) and the first law of thermodynamics',
    'エンタルピー (H) の物理的意味と等圧過程での役割を理解する': 'Understand the physical meaning of enthalpy (H) and its role in isobaric processes',
    'Helmholtz自由エネルギー (F) と等温過程での仕事を学ぶ': 'Learn about Helmholtz free energy (F) and work in isothermal processes',
    'Gibbs自由エネルギー (G) と化学平衡・相平衡への応用を理解する': 'Understand Gibbs free energy (G) and its applications to chemical and phase equilibria',
    '4つの熱力学ポテンシャルの使い分けを習得する': 'Master how to choose among the four thermodynamic potentials',
    'Pythonで熱力学関数を計算し、可視化する': 'Calculate and visualize thermodynamic functions using Python',
}

def translate_html(content):
    """Translate HTML content from Japanese to English"""

    # Update lang attribute
    content = content.replace('lang="ja"', 'lang="en"')

    # Apply direct translations
    for jp, en in translations.items():
        content = content.replace(jp, en)

    return content

def main():
    source_dir = Path('/Users/yusukehashimoto/Documents/pycharm/AI_Homepage/wp/knowledge/jp/FM/equilibrium-thermodynamics')
    target_dir = Path('/Users/yusukehashimoto/Documents/pycharm/AI_Homepage/wp/knowledge/en/FM/equilibrium-thermodynamics')

    files_to_translate = [
        'index.html',
        'chapter-1.html',
        'chapter-2.html',
        'chapter-3.html',
        'chapter-4.html',
        'chapter-5.html'
    ]

    for filename in files_to_translate:
        source_file = source_dir / filename
        target_file = target_dir / filename

        if not source_file.exists():
            print(f"Skipping {filename} - source not found")
            continue

        print(f"Processing {filename}...")

        with open(source_file, 'r', encoding='utf-8') as f:
            content = f.read()

        translated = translate_html(content)

        with open(target_file, 'w', encoding='utf-8') as f:
            f.write(translated)

        print(f"  Completed {filename}")

    print("\nTranslation complete!")

if __name__ == '__main__':
    main()
