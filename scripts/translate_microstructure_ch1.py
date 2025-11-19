#!/usr/bin/env python3
"""
Translate materials-microstructure-introduction chapter-1 from Japanese to English
Preserves all HTML structure, code blocks, and mathematical expressions
"""

import re
from pathlib import Path
from typing import Dict, Tuple

# Translation mappings
TRANSLATIONS = {
    # Meta information
    'lang="ja"': 'lang="en"',
    '読了時間': 'Reading time',
    '難易度': 'Difficulty',
    'コード例': 'Code examples',
    '中級': 'Intermediate',
    '個': 'examples',
    '分': 'min',

    # Navigation
    'AI寺子屋トップ': 'AI Terakoya Top',
    '材料組織学入門': 'Introduction to Materials Microstructure',
    '第1章': 'Chapter 1',

    # Title and subtitle
    '第1章：結晶粒と粒界の基礎': 'Chapter 1: Fundamentals of Grain Structures and Grain Boundaries',
    '組織制御による材料強化の原理': 'Principles of Material Strengthening through Microstructure Control',

    # Chapter description
    '結晶粒（grain）は多結晶材料の基本構成単位であり、その大きさと分布が材料の機械的性質を大きく左右します。この章では、結晶粒と粒界の基礎概念、Hall-Petch関係による強化メカニズム、EBSD（電子後方散乱回折）解析の基礎を学び、組織制御による材料設計の基盤を築きます。':
        'Grains are the fundamental structural units of polycrystalline materials, and their size and distribution significantly affect the mechanical properties of materials. In this chapter, we will learn the basic concepts of grains and grain boundaries, strengthening mechanisms through the Hall-Petch relationship, fundamentals of EBSD (Electron Backscatter Diffraction) analysis, and establish a foundation for materials design through microstructure control.',

    # Learning objectives
    '学習目標': 'Learning Objectives',
    'この章を読むことで、以下を習得できます：': 'By reading this chapter, you will be able to:',
    '結晶粒と粒界の定義と種類を説明できる': 'Explain the definitions and types of grains and grain boundaries',
    'Hall-Petch関係を用いて粒径と強度の関係を定量的に理解できる': 'Quantitatively understand the relationship between grain size and strength using the Hall-Petch relationship',
    '粒界の結晶学的分類（角度、CSL理論）を理解できる': 'Understand crystallographic classification of grain boundaries (angle, CSL theory)',
    'Pythonで粒径分布の統計解析ができる': 'Perform statistical analysis of grain size distribution using Python',
    '粒成長のシミュレーションを実装できる': 'Implement simulations of grain growth',
    'EBSD データの基本的な処理と可視化ができる': 'Perform basic processing and visualization of EBSD data',
    '組織-特性相関を定量的に評価できる': 'Quantitatively evaluate microstructure-property correlations',

    # Section 1.1
    '1.1 結晶粒とは何か': '1.1 What are Grains?',
    '多結晶材料の構造': 'Structure of Polycrystalline Materials',
    '実用材料の多くは<strong>多結晶体（polycrystalline material）</strong>です。多結晶体は、結晶方位が異なる多数の小さな結晶（<strong>結晶粒、grain</strong>）が集まって形成されています。':
        'Most practical materials are <strong>polycrystalline materials</strong>. Polycrystalline materials are formed by the assembly of numerous small crystals (<strong>grains</strong>) with different crystallographic orientations.',

    '<strong>結晶粒（grain）</strong>とは、内部で原子配列が一様で連続的な結晶領域のことです。隣接する結晶粒とは結晶方位が異なり、その境界を<strong>粒界（grain boundary）</strong>と呼びます。':
        'A <strong>grain</strong> is a crystalline region with a uniform and continuous atomic arrangement internally. It has a different crystallographic orientation from adjacent grains, and the boundary is called a <strong>grain boundary</strong>.',

    # Mermaid diagram
    '単結晶': 'Single Crystal',
    '結晶方位が1つ': 'One crystallographic orientation',
    '完全に一様な原子配列': 'Completely uniform atomic arrangement',
    '多結晶': 'Polycrystalline',
    '多数の結晶粒': 'Multiple grains',
    'それぞれ異なる結晶方位': 'Each with different crystallographic orientation',
    '粒界で区切られる': 'Separated by grain boundaries',

    # Importance of grains
    '結晶粒の重要性': 'Importance of Grains',
    '結晶粒の大きさ（<strong>粒径、grain size</strong>）は、材料の機械的性質に決定的な影響を与えます：':
        'The size of grains (<strong>grain size</strong>) has a decisive influence on the mechanical properties of materials:',

    '<strong>細粒化（微細化）</strong> → 強度・硬度の向上（Hall-Petch関係）': '<strong>Grain refinement</strong> → Improvement in strength and hardness (Hall-Petch relationship)',
    '<strong>粗大化</strong> → 延性の向上、クリープ抵抗の低下': '<strong>Grain coarsening</strong> → Improvement in ductility, reduction in creep resistance',
    '<strong>粒界の性質</strong> → 腐食抵抗、拡散速度、破壊挙動に影響': '<strong>Grain boundary properties</strong> → Affect corrosion resistance, diffusion rate, and fracture behavior',

    '実例': 'Examples',
    '<strong>自動車用鋼板</strong>: 平均粒径5-15 μm（高強度）': '<strong>Automotive steel sheets</strong>: Average grain size 5-15 μm (high strength)',
    '<strong>航空機用Al合金</strong>: 平均粒径50-100 μm（延性重視）': '<strong>Aerospace Al alloys</strong>: Average grain size 50-100 μm (ductility-focused)',
    '<strong>ナノ結晶材料</strong>: 平均粒径 &lt; 100 nm（超高強度）': '<strong>Nanocrystalline materials</strong>: Average grain size &lt; 100 nm (ultra-high strength)',

    # Grain size measurement
    '粒径の測定方法': 'Grain Size Measurement Methods',
    '粒径は、以下のいずれかの方法で定量化されます：': 'Grain size is quantified by one of the following methods:',

    '1. 平均線分法（Line Intercept Method）': '1. Line Intercept Method',
    '組織写真上に任意の直線を引き、粒界との交点数から計算します。': 'Draw an arbitrary straight line on the microstructure image and calculate from the number of intersections with grain boundaries.',
    'ここで、$\\bar{d}$は平均粒径、$L$は線分の長さ、$N$は粒界交点数です。': 'where $\\bar{d}$ is the average grain size, $L$ is the length of the line segment, and $N$ is the number of grain boundary intersections.',

    '2. 面積法（Planimetric Method）': '2. Planimetric Method',
    '画像解析で各結晶粒の面積を測定し、円相当直径を計算します。': 'Measure the area of each grain by image analysis and calculate the equivalent circle diameter.',
    'ここで、$d_i$は結晶粒$i$の円相当直径、$A_i$はその面積です。': 'where $d_i$ is the equivalent circle diameter of grain $i$, and $A_i$ is its area.',

    '3. ASTM粒度番号（ASTM Grain Size Number）': '3. ASTM Grain Size Number',
    '標準チャートと比較する方法です。粒度番号$G$と平均粒径の関係：': 'A method of comparison with standard charts. Relationship between grain size number $G$ and average grain size:',
    'ここで、$N$は1平方インチ（645 mm²）あたりの結晶粒数です。': 'where $N$ is the number of grains per square inch (645 mm²).',
}

def read_file_content(file_path: Path) -> str:
    """Read entire file content"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

def count_japanese_characters(text: str) -> int:
    """Count Japanese characters (hiragana, katakana, kanji)"""
    japanese_pattern = re.compile(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF]')
    return len(japanese_pattern.findall(text))

def translate_content(content: str, translations: Dict[str, str]) -> str:
    """Apply translations to content"""
    result = content

    # Sort by length (longest first) to avoid partial replacements
    sorted_translations = sorted(translations.items(), key=lambda x: len(x[0]), reverse=True)

    for japanese, english in sorted_translations:
        result = result.replace(japanese, english)

    return result

def main():
    source_path = Path('/Users/yusukehashimoto/Documents/pycharm/AI_Homepage/wp/knowledge/jp/MS/materials-microstructure-introduction/chapter-1.html')
    target_path = Path('/Users/yusukehashimoto/Documents/pycharm/AI_Homepage/wp/knowledge/en/MS/materials-microstructure-introduction/chapter-1.html')

    # Ensure target directory exists
    target_path.parent.mkdir(parents=True, exist_ok=True)

    # Read source content
    print(f"Reading source file: {source_path}")
    original_content = read_file_content(source_path)

    # Count original Japanese characters
    original_jp_count = count_japanese_characters(original_content)
    print(f"Original Japanese character count: {original_jp_count}")

    # This is a partial translation - need to read more sections
    print("\nThis script contains partial translations.")
    print("Full translation requires reading all sections of the file.")
    print("\nPlease run the complete translation process.")

if __name__ == '__main__':
    main()
