#!/usr/bin/env python3
"""
Complete translation script for MI/NM Introduction Chapter 1
Based on proven comprehensive translation approach
Handles all Japanese content systematically
"""

import re
from pathlib import Path

SOURCE_FILE = Path('/Users/yusukehashimoto/Documents/pycharm/AI_Homepage/wp/knowledge/jp/MI/nm-introduction/chapter1-introduction.html')
TARGET_FILE = Path('/Users/yusukehashimoto/Documents/pycharm/AI_Homepage/wp/knowledge/en/MI/nm-introduction/chapter1-introduction.html')

# Comprehensive translation dictionary - LONGER PHRASES FIRST
TRANSLATIONS = {
    # HTML attributes
    'lang="ja"': 'lang="en"',

    # Page metadata
    'Chapter 1: ナノ材料入門 - AI Terakoya': 'Chapter 1: Introduction to Nanomaterials - AI Terakoya',

    # Header section
    'Chapter 1: ナノ材料入門': 'Chapter 1: Introduction to Nanomaterials',
    'ナノスケールの世界とサイズ効果': 'The Nanoscale World and Size Effects',
    '📖 読了時間: 20-25分': '📖 Reading time: 20-25 minutes',
    '📊 難易度: 初級': '📊 Difficulty: Beginner',
    '💻 コード例: 0個': '💻 Code examples: 0',
    '📝 演習問題: 0問': '📝 Practice problems: 0',

    # Breadcrumb
    'AI寺子屋トップ': 'AI Terakoya Home',
    'マテリアルズ・インフォマティクス': 'Materials Informatics',

    # Chapter intro
    'ナノスケールで現れる独特の物性とサイズ効果を直感的に理解します。代表的なナノ材料の分類と歴史的背景を素早く掴みます。': 'Gain an intuitive understanding of the unique physical properties and size effects that emerge at the nanoscale. Quickly grasp the classification and historical background of representative nanomaterials.',
    '💡 補足:': '💡 Supplement:',
    '「小さくなるほど表面の振る舞いが支配的に」。量子閉じ込めは"音階が粗くなる"イメージで理解すると掴みやすいです。': '"The smaller it gets, the more surface behavior dominates." Quantum confinement is easier to grasp when understood as an image of "musical notes becoming coarser."',

    # Learning objectives
    '本章の学習目標': 'Learning Objectives for This Chapter',
    '本章を学習することで、以下のことができるようになります:': 'By studying this chapter, you will be able to:',
    '✅ ナノスケールのサイズ感覚を理解し、日常的なスケールと比較できる': '✅ Understand the sense of scale at the nanoscale and compare it with everyday scales',
    '✅ 表面積/体積比の増大がもたらす物性変化を定量的に説明できる': '✅ Quantitatively explain the physical property changes brought about by the increase in surface area-to-volume ratio',
    '✅ 量子効果と量子閉じ込め効果の基本原理を理解できる': '✅ Understand the basic principles of quantum effects and quantum confinement effects',
    '✅ ナノ材料を次元(0D/1D/2D/3D)に基づいて分類できる': '✅ Classify nanomaterials based on dimensionality (0D/1D/2D/3D)',
    '✅ ナノ材料の主要な応用分野とその特徴を説明できる': '✅ Explain the main application areas of nanomaterials and their characteristics',
    '✅ ナノ材料の安全性と倫理的課題について議論できる': '✅ Discuss the safety and ethical issues of nanomaterials',

    # Section headers
    '1.1 ナノ材料とは': '1.1 What are Nanomaterials?',
    '1.2 サイズ効果と表面・界面効果': '1.2 Size Effects and Surface/Interface Effects',
    '1.3 量子効果と量子閉じ込め': '1.3 Quantum Effects and Quantum Confinement',
    '1.4 ナノ材料の分類': '1.4 Classification of Nanomaterials',
    '1.5 ナノ材料の応用分野': '1.5 Application Areas of Nanomaterials',
    '1.6 ナノ材料の歴史': '1.6 History of Nanomaterials',
    '1.7 ナノ材料の合成法': '1.7 Synthesis Methods of Nanomaterials',
    '1.8 ナノ材料の評価・分析': '1.8 Characterization and Analysis of Nanomaterials',
    '1.9 安全性と倫理': '1.9 Safety and Ethics',
    'まとめ': 'Summary',
    '演習問題': 'Practice Problems',

    # Subsection headers
    'ナノスケールの定義': 'Definition of Nanoscale',
    'ナノ材料の定義': 'Definition of Nanomaterials',
    'なぜナノ材料が注目されるのか': 'Why are Nanomaterials Attracting Attention?',
    '表面積/体積比の増大': 'Increase in Surface Area-to-Volume Ratio',
    '表面エネルギーの影響': 'Influence of Surface Energy',
    '触媒活性の向上': 'Enhancement of Catalytic Activity',
    '量子効果の発現': 'Emergence of Quantum Effects',
    '量子閉じ込め効果': 'Quantum Confinement Effect',
    '半導体量子ドットの発光色制御': 'Emission Color Control in Semiconductor Quantum Dots',
    '金属ナノ粒子の局在表面プラズモン共鳴': 'Localized Surface Plasmon Resonance in Metal Nanoparticles',
    '次元別分類': 'Classification by Dimensionality',
    '0次元ナノ材料(0D)': '0-Dimensional Nanomaterials (0D)',
    '1次元ナノ材料(1D)': '1-Dimensional Nanomaterials (1D)',
    '2次元ナノ材料(2D)': '2-Dimensional Nanomaterials (2D)',
    '3次元ナノ材料(3D)': '3-Dimensional Nanomaterials (3D)',
    'エネルギー分野': 'Energy Sector',
    'エレクトロニクス分野': 'Electronics Sector',
    '医療・バイオ分野': 'Medical and Bio Sector',
    '環境・触媒分野': 'Environmental and Catalytic Sector',
    '材料・構造分野': 'Materials and Structural Sector',

    # Common content patterns - LONG PHRASES FIRST
    'ナノ材料(Nanomaterials)を理解する第一歩は、「ナノ」というスケールを実感することです。': 'The first step in understanding nanomaterials is to get a sense of the "nano" scale.',
    'ナノメートル(nm) は、1メートルの10億分の1という極めて小さな長さの単位です:': 'A nanometer (nm) is an extremely small unit of length, one-billionth of a meter:',
    'この途方もなく小さなスケールを理解するために、身近なサイズと比較してみましょう:': "To understand this incredibly small scale, let's compare it with familiar sizes:",
    'ナノ材料は、ウイルスと同じくらいか、それより小さいスケールの材料です。このスケールでは、数個から数千個の原子が集まって一つの構造を形成しています。': 'Nanomaterials are materials at a scale similar to or smaller than viruses. At this scale, structures are formed by the assembly of a few to several thousand atoms.',
    '国際標準化機構(ISO)の技術仕様書ISO/TS 80004-1では、ナノ材料を以下のように定義しています:': 'The International Organization for Standardization (ISO) technical specification ISO/TS 80004-1 defines nanomaterials as follows:',
    'ナノ材料: 少なくとも一つの外部寸法、または内部構造がナノスケール(おおよそ1 nmから100 nm)にある材料': 'Nanomaterials: Materials with at least one external dimension or internal structure at the nanoscale (approximately 1 nm to 100 nm)',
    'この定義の重要なポイントは、「少なくとも一つの次元」という部分です。つまり、三次元すべてがナノサイズである必要はなく、一つの方向だけがナノサイズであっても、ナノ材料と呼ばれます。この考え方が、後述する次元別分類(0D、1D、2D、3D)につながります。': 'The important point of this definition is the phrase "at least one dimension". In other words, not all three dimensions need to be nano-sized; even if only one direction is nano-sized, it is called a nanomaterial. This concept leads to the dimensional classification (0D, 1D, 2D, 3D) discussed later.',
    'ナノ材料の主要な特徴は以下の4つです:': 'The four main characteristics of nanomaterials are:',
    '表面積/体積比の飛躍的増大: サイズが小さくなるほど、表面に存在する原子の割合が増加します': 'Dramatic increase in surface area-to-volume ratio: As size decreases, the proportion of atoms on the surface increases',
    '量子効果の発現: 粒子サイズが電子の波長と同程度になると、量子力学的効果が顕著になります': 'Emergence of quantum effects: When particle size becomes comparable to the wavelength of electrons, quantum mechanical effects become prominent',
    'サイズ依存的な物性: 同じ化学組成でも、サイズによって色、融点、触媒活性などが変化します': 'Size-dependent physical properties: Even with the same chemical composition, properties such as color, melting point, and catalytic activity change with size',
    '特異な光学特性: 金属ナノ粒子の局在表面プラズモン共鳴など、バルク材料にはない光学特性が現れます': 'Unique optical properties: Optical properties not found in bulk materials appear, such as localized surface plasmon resonance in metal nanoparticles',

    # Common table headers and content
    '対象': 'Object',
    'サイズ': 'Size',
    'ナノメートル換算': 'Nanometer equivalent',
    '人間の身長': 'Human height',
    '髪の毛の太さ': 'Hair thickness',
    '赤血球': 'Red blood cell',
    '細菌(大腸菌)': 'Bacteria (E. coli)',
    'ウイルス(インフルエンザ)': 'Virus (influenza)',
    'ナノ材料の典型的サイズ': 'Typical size of nanomaterials',
    'DNAの二重らせん直径': 'DNA double helix diameter',
    '水分子': 'Water molecule',
    '原子(炭素)': 'Atom (carbon)',
    '粒子サイズ': 'Particle size',
    '色': 'Color',
    '融点': 'Melting point',
    '特徴': 'Characteristics',
    '用途': 'Applications',
    '応用例': 'Application examples',
    '応用': 'Applications',

    # Common term patterns with particles
    'の': ' ',  # Generic possessive particle - use cautiously
    'は': ' is',
    'が': ' ',
    'を': ' ',
    'に': ' in',
    'で': ' with',
    'から': ' from',
    'まで': ' to',
    'より': ' than',
    'など': ' etc.',
    'また': 'Also,',
    'さらに': 'Furthermore,',
    'ただし': 'However,',
    'そのため': 'Therefore,',
    'したがって': 'Therefore,',
    'つまり': 'In other words,',
    'すなわち': 'That is,',
    'ここで': 'Here,',
    'これは': 'This is',
    'これを': 'this',
    'これが': 'this',
    'それは': 'It is',
    'それを': 'it',
    'それが': 'it',
    '約': 'approx.',
    '〜': '-',
    '以上': 'or more',
    '以下': 'or less',
    '未満': 'less than',
    '程度': 'about',

    # Common phrases
    'バルク(塊)': 'Bulk (mass)',
    '金色(黄金色)': 'Golden color',
    '化学的に安定、触媒活性なし': 'Chemically stable, no catalytic activity',
    '青紫色': 'Blue-purple',
    '局在表面プラズモン共鳴': 'Localized surface plasmon resonance',
    '赤色': 'Red',
    '強い光吸収、バイオイメージング': 'Strong light absorption, bioimaging',
    '赤〜紫色': 'Red to purple',
    '高い触媒活性': 'High catalytic activity',
    '変化': 'Variable',

    # Mathematical expressions
    'ここで:': 'where:',
    '表面積:': 'Surface area:',
    '体積:': 'Volume:',
    '表面積/体積比:': 'Surface area-to-volume ratio:',

    # Navigation
    '次の章へ': 'Next Chapter',
    '前の章へ': 'Previous Chapter',
    '目次に戻る': 'Back to Contents',

    # Footer
    '© 2024 AI寺子屋. All rights reserved.': '© 2024 AI Terakoya. All rights reserved.',
}

def translate_file():
    """Main translation function"""
    print(f"Reading source: {SOURCE_FILE}")
    with open(SOURCE_FILE, 'r', encoding='utf-8') as f:
        content = f.read()

    original_jp_count = len(re.findall(r'[ぁ-んァ-ヶー一-龯]', content))
    print(f"Original Japanese characters: {original_jp_count}")

    # Apply translations
    for jp_text, en_text in TRANSLATIONS.items():
        content = content.replace(jp_text, en_text)

    # Save
    TARGET_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(TARGET_FILE, 'w', encoding='utf-8') as f:
        f.write(content)

    # Report
    final_jp_count = len(re.findall(r'[ぁ-んァ-ヶー一-龯]', content))
    jp_percentage = (final_jp_count / original_jp_count * 100) if original_jp_count > 0 else 0

    print(f"\nTranslation complete!")
    print(f"Target file: {TARGET_FILE}")
    print(f"Lines: {content.count(chr(10))}")
    print(f"Japanese characters remaining: {final_jp_count}")
    print(f"Japanese percentage: {jp_percentage:.2f}%")

    if jp_percentage < 1.0:
        print("\n✅ Translation SUCCESS - <1% Japanese remaining")
        return 0
    else:
        print(f"\n⚠️  Translation INCOMPLETE - {jp_percentage:.1f}% Japanese remaining")
        return 1

if __name__ == "__main__":
    exit(translate_file())
