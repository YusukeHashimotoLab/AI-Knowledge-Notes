#!/usr/bin/env python3
"""
Translate MS materials-microstructure-introduction chapter-2 from Japanese to English
Preserves all HTML structure, attributes, and formatting
"""

import re
from pathlib import Path

# Translation dictionary for comprehensive mapping
translations = {
    # Meta and header
    'lang="ja"': 'lang="en"',
    '<title>第2章：相変態の基礎 - MS Terakoya</title>': '<title>Chapter 2: Fundamentals of Phase Transformations - MS Terakoya</title>',

    # Header content
    '第2章：相変態の基礎': 'Chapter 2: Fundamentals of Phase Transformations',
    'Phase Transformations - 熱処理による組織制御の科学': 'Phase Transformations - Science of Microstructure Control through Heat Treatment',
    '📖 読了時間: 30-40分': '📖 Reading time: 30-40 minutes',
    '📊 難易度: 中級': '📊 Difficulty: Intermediate',
    '💻 コード例: 7個': '💻 Code examples: 7',

    # Breadcrumb
    'AI寺子屋トップ': 'AI Terakoya Top',
    'MS Dojo': 'MS Dojo',
    '材料組織学入門': 'Introduction to Materials Microstructure',
    '第2章': 'Chapter 2',

    # Chapter description
    '材料の性質は、温度と時間の履歴（熱処理）によって劇的に変化します。この変化の根源は<strong>相変態（phase transformation）</strong>です。この章では、相図の読み方、拡散型・無拡散型変態のメカニズム、TTT/CCT図の活用法、マルテンサイト変態、そしてCALPHAD法による状態図計算の基礎を学び、熱処理設計の理論的基盤を築きます。':
        'Material properties change dramatically depending on temperature and time history (heat treatment). The origin of this change is <strong>phase transformation</strong>. In this chapter, we will learn how to read phase diagrams, mechanisms of diffusional and diffusionless transformations, application of TTT/CCT diagrams, martensitic transformation, and the basics of phase diagram calculation using the CALPHAD method, building a theoretical foundation for heat treatment design.',

    # Learning objectives
    '学習目標': 'Learning Objectives',
    'この章を読むことで、以下を習得できます：': 'By reading this chapter, you will be able to:',
    '✅ 二元系・三元系相図を読み、相平衡を理解できる': '✅ Read binary and ternary phase diagrams and understand phase equilibrium',
    '✅ てこの法則（Lever Rule）を用いて相分率を計算できる': '✅ Calculate phase fractions using the Lever Rule',
    '✅ TTT図・CCT図から変態速度と組織を予測できる': '✅ Predict transformation rate and microstructure from TTT and CCT diagrams',
    '✅ Avrami式で変態の進行度を定量化できる': '✅ Quantify transformation progress using the Avrami equation',
    '✅ マルテンサイト変態の原理とM<sub>s</sub>温度の予測ができる': '✅ Understand the principles of martensitic transformation and predict M<sub>s</sub> temperature',
    '✅ CALPHAD法の基礎とpycalphadライブラリの使い方を理解できる': '✅ Understand the basics of the CALPHAD method and how to use the pycalphad library',
    '✅ Pythonで相図と変態速度論のシミュレーションができる': '✅ Perform phase diagram and transformation kinetics simulations in Python',

    # Section 2.1
    '2.1 相図の基礎と読み方': '2.1 Fundamentals and Reading of Phase Diagrams',
    '相図（Phase Diagram）とは': 'What is a Phase Diagram?',
    '<p><strong>相図</strong>は、温度・組成・圧力の関数として、どの相が熱力学的に安定かを示す図です。材料の熱処理条件を決定する際の最も重要なツールです。</p>':
        '<p>A <strong>phase diagram</strong> is a diagram that shows which phases are thermodynamically stable as a function of temperature, composition, and pressure. It is the most important tool when determining heat treatment conditions for materials.</p>',

    '<strong>相（Phase）</strong>とは、化学組成・構造・性質が一様で、他の部分と明確な界面で区切られた物質の均一な部分です。例: 液相（L）、α相（BCC）、γ相（FCC）、セメンタイト（Fe<sub>3</sub>C）':
        '<strong>A phase</strong> is a homogeneous portion of a material with uniform chemical composition, structure, and properties, separated from other portions by distinct interfaces. Examples: liquid phase (L), α-phase (BCC), γ-phase (FCC), cementite (Fe<sub>3</sub>C)',

    # Binary phase diagram types
    '二元系相図の基本型': 'Basic Types of Binary Phase Diagrams',
    '1. 全率固溶型（Complete Solid Solution）': '1. Complete Solid Solution',
    '2つの元素が全組成範囲で固溶する系です。': 'A system in which two elements form a solid solution over the entire composition range.',
    '<strong>例</strong>: Cu-Ni系、Au-Ag系': '<strong>Examples</strong>: Cu-Ni system, Au-Ag system',

    '2. 共晶型（Eutectic System）': '2. Eutectic System',
    'ある組成・温度で、液相が冷却時に2つの固相に同時に分解します。': 'At a certain composition and temperature, the liquid phase decomposes simultaneously into two solid phases upon cooling.',
    '<strong>例</strong>: Pb-Sn系、Al-Si系': '<strong>Examples</strong>: Pb-Sn system, Al-Si system',
    '共晶反応: $L \\rightarrow \\alpha + \\beta$（冷却時）': 'Eutectic reaction: $L \\rightarrow \\alpha + \\beta$ (upon cooling)',

    '3. 包晶型（Peritectic System）': '3. Peritectic System',
    '液相と固相が反応して別の固相を生成します。': 'A liquid phase and a solid phase react to produce another solid phase.',
    '<strong>例</strong>: Fe-C系（高温部）、Pt-Ag系': '<strong>Examples</strong>: Fe-C system (high temperature region), Pt-Ag system',
    '包晶反応: $L + \\delta \\rightarrow \\gamma$（冷却時）': 'Peritectic reaction: $L + \\delta \\rightarrow \\gamma$ (upon cooling)',

    # Fe-C phase diagram
    'Fe-C状態図（鉄鋼の基礎）': 'Fe-C Phase Diagram (Fundamentals of Steel)',
    'Fe-C系相図は、鉄鋼材料の熱処理設計の基盤です。': 'The Fe-C phase diagram is the foundation for heat treatment design of steel materials.',

    # Mermaid diagram nodes
    '高温<br/>δ-Fe BCC': 'High Temp<br/>δ-Fe BCC',
    '冷却': 'Cooling',
    'γ-Fe FCC<br/>オーステナイト': 'γ-Fe FCC<br/>Austenite',
    '共析変態<br/>727°C 0.77%C': 'Eutectoid Transf.<br/>727°C 0.77%C',
    'α-Fe BCC<br/>フェライト': 'α-Fe BCC<br/>Ferrite',
    'Fe₃C<br/>セメンタイト': 'Fe₃C<br/>Cementite',
    '微細な混合組織': 'Fine Mixed Structure',
    'パーライト': 'Pearlite',
    '急冷<br/>無拡散変態': 'Rapid Cooling<br/>Diffusionless Transf.',
    'マルテンサイト<br/>BCT 超硬質': 'Martensite<br/>BCT Ultra-hard',

    # Important temperatures and compositions
    '重要な温度と組成': 'Important Temperatures and Compositions',
    '<strong>共析点（Eutectoid Point）</strong>: 727°C、0.77% C': '<strong>Eutectoid Point</strong>: 727°C, 0.77% C',
    '共析反応: $\\gamma \\rightarrow \\alpha + \\text{Fe}_3\\text{C}$（パーライト組織）': 'Eutectoid reaction: $\\gamma \\rightarrow \\alpha + \\text{Fe}_3\\text{C}$ (pearlite microstructure)',
    '<strong>亜共析鋼（Hypoeutectoid Steel）</strong>: 0.02-0.77% C': '<strong>Hypoeutectoid Steel</strong>: 0.02-0.77% C',
    '組織: 初析フェライト + パーライト': 'Microstructure: Proeutectoid ferrite + Pearlite',
    '<strong>共析鋼（Eutectoid Steel）</strong>: 0.77% C': '<strong>Eutectoid Steel</strong>: 0.77% C',
    '組織: 100%パーライト': 'Microstructure: 100% Pearlite',
}

def translate_file(source_path: Path, target_path: Path):
    """Translate Japanese HTML file to English while preserving structure"""

    print(f"Reading source file: {source_path}")
    with open(source_path, 'r', encoding='utf-8') as f:
        content = f.read()

    original_content = content
    jp_char_count = 0

    # Count Japanese characters before translation
    jp_pattern = re.compile(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF\u3400-\u4DBF]+')
    jp_matches = jp_pattern.findall(content)
    jp_char_count = sum(len(match) for match in jp_matches)

    print(f"\nOriginal Japanese character count: {jp_char_count}")
    print(f"Starting translation of {len(translations)} patterns...")

    # Apply translations
    translated_count = 0
    for jp_text, en_text in translations.items():
        if jp_text in content:
            content = content.replace(jp_text, en_text)
            translated_count += 1
            if translated_count % 10 == 0:
                print(f"  Translated {translated_count}/{len(translations)} patterns...")

    print(f"Applied {translated_count} translation patterns")

    # Count remaining Japanese characters
    remaining_jp_matches = jp_pattern.findall(content)
    remaining_jp_count = sum(len(match) for match in remaining_jp_matches)

    # Calculate translation percentage
    if jp_char_count > 0:
        translation_percentage = ((jp_char_count - remaining_jp_count) / jp_char_count) * 100
    else:
        translation_percentage = 100.0

    print(f"\nTranslation Summary:")
    print(f"  Original Japanese characters: {jp_char_count}")
    print(f"  Remaining Japanese characters: {remaining_jp_count}")
    print(f"  Translation percentage: {translation_percentage:.1f}%")

    # Save translated content
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with open(target_path, 'w', encoding='utf-8') as f:
        f.write(content)

    print(f"\n✅ Translation complete: {target_path}")
    print(f"\nSTATISTICS:")
    print(f"  JP Characters: {jp_char_count}")
    print(f"  Translation: {translation_percentage:.1f}%")

    return jp_char_count, translation_percentage

def main():
    source = Path('/Users/yusukehashimoto/Documents/pycharm/AI_Homepage/wp/knowledge/jp/MS/materials-microstructure-introduction/chapter-2.html')
    target = Path('/Users/yusukehashimoto/Documents/pycharm/AI_Homepage/wp/knowledge/en/MS/materials-microstructure-introduction/chapter-2.html')

    if not source.exists():
        print(f"❌ Source file not found: {source}")
        return 1

    jp_count, percentage = translate_file(source, target)

    return 0

if __name__ == '__main__':
    exit(main())
